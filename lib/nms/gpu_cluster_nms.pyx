# --------------------------------------------------------
# gpu-cluster-nms
# Written by Chen Xu
# --------------------------------------------------------

import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_cluster_nms.hpp":
    void _cluster_nms(np.int32_t*, int*, np.float32_t*, int, int, float, int)

def gpu_cluster_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh,
            np.int32_t device_id=0):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = dets.shape[1]
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] \
        keep = np.zeros(boxes_num, dtype=np.int32)
    _cluster_nms(&keep[0],&num_out,&dets,boxes_num,boxes_dim,thresh,device_id)
    keep = keep[:num_out]
    return list(order[keep])
