// ------------------------------------------------------------------
// cluster based nms kernel
// Written by Chen Xu
// ------------------------------------------------------------------

#include "gpu_cluster_nms.hpp"
#include <iostream>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void _cluster_nms_kernel(const int n_boxes, const float thresh,
    const float *dev_boxes, unsigned long long *dev_mask)
{
    const int row_size = min(n_boxes - blockIdx.x * threadsPerBlock, threadsPerBlock);
    const int col_block = DIVUP(n_boxes, threadsPerBlock);
    int box_idx = (blockIdx.x * threadPerBlock + threadIdx.x);
    float cur[5];
    if(box_idx < n_boxes)
    {
        cur[0] = dev_boxes[box_idx * 5];
        cur[1] = dev_boxes[box_idx * 5 + 1];
        cur[2] = dev_boxes[box_idx * 5 + 2];
        cur[3] = dev_boxes[box_idx * 5 + 3];
        cur[4] = dev_boxes[box_idx * 5 + 4];
    }
    extern __shared__ float* cache[];
    for(int i = 0;i<col_block;++i)
    {
        int baseoffset = i * threadsPerBlock;
        int col_size = min(n_boxes - baseoffset, threadsPerBlock);
        if(baseoffset + threadIdx.x < n_boxes)
        {
            cache[threadIdx.x * 5] = dev_boxes[(baseoffset + threadIdx.x) * 5];
            cache[threadIdx.x * 5 + 1] = dev_boxes[(baseoffset + threadIdx.x) * 5 + 1];
            cache[threadIdx.x * 5 + 2] = dev_boxes[(baseoffset + threadIdx.x) * 5 + 2];
            cache[threadIdx.x * 5 + 3] = dev_boxes[(baseoffset + threadIdx.x) * 5 + 3];
            cache[threadIdx.x * 5 + 4] = dev_boxes[(baseoffset + threadIdx.x) * 5 + 4];
        }
        __syncthreads();
        if(threadIdx.x < row_size)
        {
            float scoreadj = 0;
            for(int j = 0;j < col_size;++j)
            {
                if(box_idx == baseoffset + threadIdx.x)
                    continue;
                float iou = devIoU(cache + j * 5, cur);
                if(iou > 0))
                {
                    if(cache[j * 5 + 4] > cur[4])
                    {
                        // linear supppression
                        cur[4] = cur[4] * iou;
                    }
                }
            }
        }
    }
    __shared__ int res[threadPerBlock];
    // discard if score is too low
    if(cur[4] < thresh)
    {
        res[threadIdx.x] = 1;
    }
    else
    {
        res[threadIdx.x] = 0;
    }
    __syncthreads();
    // encode
    if(threadIdx.x == 0)
    {
        unsigned long long res_ull = 0;
        for(int i = 0; i < threadPerBlock; ++i)
        {
            res_ull |= 1ULL << i;
        }
    }
    dev_mask[blockIdx.x] = res_ull;
}

void _cluster_nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
    int boxes_dim, float nms_ov, int device_id)
{
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device == device_id) {
        return;
    }
    CUDA_CHECK(cudaSetDevice(device_id));

    float* boxes_dev = NULL;
    unsigned long long* mask_dev = NULL;

    // how many blocks we will have
    const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

    CUDA_CHECK(cudaMalloc(&boxes_dev,
                          boxes_num * boxes_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(boxes_dev,
                          boxes_host,
                          boxes_num * boxes_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    // result
    CUDA_CHECK(cudaMalloc(&mask_dev, col_blocks * sizeof(unsigned long long)));

    dim3 blocks(col_blocks);
    dim3 threads(threadsPerBlock);
    _cluster_nms_kernel<<<blocks, threads, sizeof(float) * threadsPerBlock * 5>>>(boxes_num,
                                  nms_ov,
                                  boxes_dev,
                                  mask_dev);

    std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
    CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * col_blocks,
                        cudaMemcpyDeviceToHost));

    int num_to_keep = 0;
    for (int i = 0; i < boxes_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(mask_host[nblock] & (1ULL << inblock))) {
            keep_out[num_to_keep++] = i;
        }
    }
    *num_out = num_to_keep;

    CUDA_CHECK(cudaFree(boxes_dev));
    CUDA_CHECK(cudaFree(mask_dev));
}

