cimport cython
import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] bbox_overlaps_op(np.ndarray[np.float32_t, ndim=2] boxes, np.ndarray[np.float32_t, ndim=2] query_boxes):
    cdef int N = boxes.shape[0]
    cdef int K = query_boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] overlaps = np.zeros((N, K), dtype=np.float32)
    cdef np.float32_t iw, ih, box_area, inter
    cdef int n, k

    for n in range(N):
        box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)

        for k in range(K):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    inter = iw * ih

                    overlaps[n, k] = inter / (box_area + (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1) - inter)

def bbox_overlaps(np.ndarray[np.float32_t, ndim=2] boxes, np.ndarray[np.float32_t, ndim=2] query_boxes):
    return bbox_overlaps_op(boxes, query_boxes)
