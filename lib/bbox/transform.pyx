cimport cython
import numpy as np
cimport numpy as np
from libc.math import exp, log

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] bbox_transform_op(np.ndarray[np.float32_t, ndim=2] ex_rois, np.ndarray[np.float32_t, ndim=2] gt_rois):
    """Compute targets for anchors and proposals
    Params:
        ex_rois (fixed anchors) [x1, y1, x2, y2]
        gt_rois (groundtruths) [x1, y1, x2, y2]
    Return:
        targets [tx, ty, tw, th]
    """
    cdef int N = ex_rois.shape[0]
    cdef int n    
    cdef np.ndarray[np.float32_t, ndim=2] targets = np.zeros((N, 4), dtype=np.float32)
    cdef np.float32_t ex_width, ex_height, ex_ctr_x, ex_ctr_y
    cdef np.float32_t gt_width, gt_height, gt_ctr_x, gt_ctr_y

    for n in range(N):
        ex_width = ex_rois[n, 2] - ex_rois[n, 0] + 1
        ex_height = ex_rois[n, 3] - ex_rois[n, 1] + 1
        ex_ctr_x = ex_rois[n, 0] + 0.5 * ex_width
        ex_ctr_y = ex_rois[n, 1] + 0.5 * ex_height

        gt_width = gt_rois[n, 2] - gt_rois[n, 0] + 1
        gt_height = gt_rois[n, 3] - gt_rois[n, 1] + 1
        gt_ctr_x = gt_rois[n, 0] + 0.5 * gt_width
        gt_ctr_y = gt_rois[n, 1] + 0.5 * gt_height

        targets[n, 0] = (gt_ctr_x - ex_ctr_x) / ex_width
        targets[n, 1] = (gt_ctr_y - ex_ctr_y) / ex_height
        targets[n, 2] = log(gt_width / ex_width)
        targets[n, 3] = log(gt_height / ex_height)

    return targets

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] bbox_transform_inv_op(np.ndarray[np.float32_t, ndim=2] boxes, np.ndarray[np.float32_t, ndim=2] deltas):
    """Compute predictions from fixed anchors and logits
    Params:
        boxes (fixed anchors) [x1, y1, x2, y2]
        deltas (logits) [tx, ty, tw, th]
    Return:
        pred_boxes [x1, y1, x2, y2]
    """
    cdef int N = boxes.shape[0]
    cdef int n
    cdef np.ndarray[np.float32_t, ndim=2] pred_boxes = np.zeros((N, 4), dtype=np.float32)
    cdef np.float32_t box_width, box_height, box_ctr_x, box_ctr_y
    cdef np.float32_t pred_width, pred_height, pred_ctr_x, pred_ctr_y

    for n in range(N):
        box_width = boxes[n, 2] - boxes[n, 0] + 1
        box_height = boxes[n, 3] - boxes[n, 1] + 1
        box_ctr_x = boxes[n, 0] + 0.5 * box_width
        box_ctr_y = boxes[n, 1] + 0.5 * box_height

        pred_ctr_x = deltas[n, 0] * box_width + box_ctr_x
        pred_ctr_y = deltas[n, 1] * box_height + box_ctr_y
        pred_width = exp(deltas[n, 2]) * box_width
        pred_height = exp(deltas[n, 3]) * box_height

        pred_boxes[n, 0] = pred_ctr_x - 0.5 * pred_width
        pred_boxes[n, 1] = pred_ctr_y - 0.5 * pred_height
        pred_boxes[n, 2] = pred_ctr_x + 0.5 * pred_width
        pred_boxes[n, 3] = pred_ctr_y + 0.5 * pred_height 

    return pred_boxes

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] clip_boxes_op(np.ndarray[np.float32_t, ndim=2] boxes, int im_width, int im_height):
    cdef int N = boxes.shape[0]
    cdef int n
    cdef np.ndarray[np.float32_t, ndim=2] clipped_boxes = np.zeros((N, 4), dtype=np.float32)

    for n in range(N):
        clipped_boxes[n, 0] = max(min(boxes[n, 0], im_width - 1), 0)
        clipped_boxes[n, 1] = max(min(boxes[n, 1], im_height - 1), 0)
        clipped_boxes[n, 2] = max(min(boxes[n, 2], im_width - 1), 0)
        clipped_boxes[n, 3] = max(min(boxes[n, 3], im_height - 1), 0)

    return clipped_boxes

def bbox_transform(np.ndarray[np.float32_t, ndim=2] ex_rois, np.ndarray[np.float32_t, ndim=2] gt_rois):
    return bbox_transform_op(ex_rois, gt_rois)

def bbox_transform_inv(np.ndarray[np.float32_t, ndim=2] boxes, np.ndarray[np.float32_t, ndim=2] deltas):
    return bbox_transform_inv_op(boxes, deltas)

def clip_boxes(np.ndarray[np.float32_t, ndim=2] boxes, int im_width, int im_height):
    return clip_boxes_op(boxes, im_width, im_height)
