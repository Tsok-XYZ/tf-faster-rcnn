from __future__ import absolute_import, division, print_function
import numpy as np


def _whctrs(anchor):
    """Get width, height and centre of anchor
    """
    width = anchor[2] - anchor[0] + 1
    height = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + (width - 1) * 0.5
    y_ctr = anchor[1] + (height - 1) * 0.5
    return width, height, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Generate anchors with scales, ratios at (x_ctr, y_ctr)
    """
    ws = (ws[:, np.newaxis] - 1) * 0.5
    hs = (hs[:, np.newaxis] - 1) * 0.5
    anchors = np.hstack([x_ctr - ws, y_ctr - hs, x_ctr + ws, y_ctr + hs])
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate anchors with scales
    """
    width, height, x_ctr, y_ctr = _whctrs(anchor)
    anchors = _mkanchors(width*scales, height*scales, x_ctr, y_ctr)
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate anchors with ratios
    """
    width, height, x_ctr, y_ctr = _whctrs(anchor)
    size_ratios = (width * height) / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)):
    """Generate base anchors from [0, 0, base_size - 1, base_size - 1]
    """
    # [[ -84.  -40.   99.   55.]
    # [-176.  -88.  191.  103.]
    # [-360. -184.  375.  199.]
    # [ -56.  -56.   71.   71.]
    # [-120. -120.  135.  135.]
    # [-248. -248.  263.  263.]
    # [ -36.  -80.   51.   95.]
    # [ -80. -168.   95.  183.]
    # [-168. -344.  183.  359.]]
    base_anchor = np.array(
        [0, 0, base_size - 1, base_size - 1], dtype=np.float32)
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(anchor, scales)
                         for anchor in ratio_anchors])
    return anchors


def generate_shift_anchors(feat_width, feat_height, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """Generate anchors and shift on image
    """
    anchors = _generate_anchors(ratios=np.array(
        anchor_ratios, dtype=np.float32), scales=np.array(anchor_scales, dtype=np.float32))

    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()  # posible positions on feature maps

    na = anchors.shape[0]
    ns = shifts.shape[0]

    shift_anchors = anchors.reshape(
        (1, na, 4)) + shifts.reshape((1, ns, 4)).transpose((1, 0, 2))
    shift_anchors = shift_anchors.reshape(
        (na*ns, 4)).astype(np.float32, copy=False)

    return shift_anchors
