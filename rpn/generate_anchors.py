# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=np.array([4.0])):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
#    x_ctr = anchor[0] + 0.5 * (w - 1)
#    y_ctr = anchor[1] + 0.5 * (h - 1)
    x_ctr=0
    y_ctr=0
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.sqrt(size_ratios)
    hs = ws * ratios
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def calc_roi_align_shifts(search_box, roi_size, bound):
    '''
    search_boxes in raw image lvl
    shifts: roi_size*roi_size*K
    anchor with shifts should be at raw image lvl
    '''
#    print(search_box)
#    A = roi_size * roi_size

    x, y, w, h = search_box[0], search_box[1], search_box[2] - search_box[0]+1, search_box[3] - search_box[1]+1

    '''use float instead of integer quantization'''
#    x, y, w, h = x // stride, y // stride, w // stride, h // stride
#    bin_size_w = max(1, np.ceil(1.0*w / roi_size))
#    bin_size_h = max(1, np.ceil(1.0*h / roi_size))
#
#    shift_x = np.asarray([x + bin_size_w * i for i in range(roi_size)]) * stride
#    shift_y = np.asarray([y + bin_size_h * i for i in range(roi_size)]) * stride
    bin_size_w=1.0*w/roi_size[0]
    bin_size_h=1.0*h/roi_size[1]
    
    xstart=x+bin_size_w*0.5
    ystart=y+bin_size_h*0.5
    shift_x = np.asarray([xstart+bin_size_w*i for i in range(roi_size[0])])
    shift_y = np.asarray([ystart+bin_size_h*i for i in range(roi_size[1])])

    shift_x=np.minimum(bound[0]-1, shift_x)
    shift_y=np.minimum(bound[1]-1, shift_y)

    shifts = np.meshgrid(shift_x, shift_y)
    shift_ctrs = np.vstack((shifts[0].ravel(), shifts[1].ravel(), shifts[0].ravel(), \
                            shifts[1].ravel())).transpose()

#    print(shift_ctrs)
    return shift_ctrs

def gen_region_anchors(_anchors, search_boxes, bound, K=9, size=None):
    box_anchors=[]
    A=size[1]*size[0]
    for i in range(len(search_boxes)):
        shifts_ctrs = calc_roi_align_shifts(search_boxes[i], size, bound)
        anchors = _anchors.reshape((1, K, 4)) + shifts_ctrs.reshape((1, A, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((A*K, 4))
        box_anchors.append(anchors)
    return box_anchors

if __name__=='__main__':
    anchors=generate_anchors()
    print(anchors)