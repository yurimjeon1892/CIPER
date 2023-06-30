from .kitti import KITTI
from .cvusa import CVUSA
from .vigor import VIGOR

def build_dataset(mode, args):

    if args["data_name"] == 'vigor':
        dataset = VIGOR
    elif args["data_name"] == 'cvusa':
        dataset = CVUSA
    elif args["data_name"] == 'kitti':
        dataset = KITTI

    return dataset(mode, args)
