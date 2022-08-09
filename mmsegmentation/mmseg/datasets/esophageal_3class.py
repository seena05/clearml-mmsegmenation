# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Esophageal3classDataset(CustomDataset):
    """COCO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    CLASSES = (
        'background', 'cancer', 'noncancer')

    PALETTE = [[0, 0, 0], [0, 100, 255], [0, 255, 100]]

    def __init__(self, **kwargs):
        super(Esophageal3classDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)
