import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.PHASE='TRAIN'
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
__C.IMAGE_NORMALIZE= False
__C.GAIN=0.02
__C.TEST_TYPE='RPN'
__C.NMS=True

__C.NUM_CLASSES=5
__C.DATA_LOADER_METHOD='INTER_SEQ'

__C.BASIC_SIZE=16
__C.RATIOS=np.asarray([0.5,1,2])
__C.SCALES=np.asarray([2,4,8,16])
__C.DET_ROI_SIZE=14
__C.TEMP_ROI_SIZE=5
__C.FRCNN_ROI_SIZE=7

__C.CHOOSE_ANCHOR=False
__C.CHOOSE_PROPOSAL=True
__C.ANCHOR_NUM=256
__C.POS_ANCHOR_FRACTION=0.5
__C.STRIDE=8

__C.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.BBOX_NORMALIZE_MEANS = np.array([[0.0, 0.0, 0.0, 0.0]])
__C.BBOX_NORMALIZE_STDS = np.array([[0.1, 0.1, 0.2, 0.2]])
__C.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
__C.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

"""TRAIN"""
'''Fast-RCNN'''
__C.TRAIN = edict()

__C.TRAIN.SCALES = (600,)
# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000
# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 2
# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 256
# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.5
# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.0

__C.TRAIN.APPLY_SHIFT=False

'''RPN'''
__C.TRAIN.HAS_RPN = False
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_THRESH = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_THRESH = 0.3
__C.TRAIN.RPN_BBOX_THRESH=0.4
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 20000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 8000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 16


"""TEST"""
__C.TEST = edict()
__C.TEST.IMS_PER_BATCH = 1
# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)
# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3
## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.8
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 1000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16


def get_output_dir(imdb, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
