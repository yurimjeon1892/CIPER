from .kitti import KITTI
from .cvusa import CVUSA
from .vigor import VIGOR
from .ford import Ford


def build_dataset(mode, args):
    if args["data_name"] == "vigor":
        dataset = VIGOR
    elif args["data_name"] == "cvusa":
        dataset = CVUSA
    elif args["data_name"] == "kitti":
        dataset = KITTI
    elif args["data_name"] == "ford":
        dataset = Ford
    else:
        print("data name error. please check config - data_name")
        exit()

    return dataset(mode, args)
