cimport cython
import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_nms.hpp":
    void _nms(np.int32_t*, int*, np.float32_t*, int, int, float, int)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list gpu_nms_op(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh, np.int32_t device_id):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = dets.shape[1]
    cdef int num_out

    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]
    cdef np.ndarray[np.float32_t, ndim=2] sorted_dets = dets[order, :]
    cdef np.ndarray[np.int32_t, ndim=1] keep = np.zeros(boxes_num, dtype=np.int32)

    _nms(&keep[0], &num_out, &sorted_dets[0, 0], boxes_num, boxes_dim, thresh, device_id)

    keep = list(order[keep[:num_out]])

    return keep


def gpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh, np.int32_t device_id=0):
    return gpu_nms_op(dets, thresh, device_id)
