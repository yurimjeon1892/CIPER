from .ford import Ford
from .kitti import KITTI
from .kittidb import KITTIDB
from .vigor import VIGOR
from .vigordb import VIGORDB


def build_dataset(mode, args):
    if args["data_name"] == "kitti":
        dataset = KITTI
    elif args["data_name"] == "kittidb":
        dataset = KITTIDB
    elif args["data_name"] == "ford":
        dataset = Ford
    elif args["data_name"] == "vigor":
        dataset = VIGOR
    elif args["data_name"] == "vigordb":
        dataset = VIGORDB
    else:
        print("data name error. please check config - data_name")
        exit()

    return dataset(mode, args)
