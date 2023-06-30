import os
import torch
import numpy as np
import shutil
import matplotlib.pyplot as plt

import random

from PIL import Image, ImageDraw

__all__ = [
    'save_state',
    'save_image'
]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_state(save_path, model, is_best, best_metric, epoch, filename="checkpoint.pth.tar"):
    os.makedirs(save_path, exist_ok=True)
    model_state_dict = model.state_dict()
    state_dict = {
        'net': model_state_dict,
        'epoch': epoch,
        'best_metric': best_metric
    }
    torch.save(state_dict, os.path.join(save_path, filename))
    shutil.copyfile(
        os.path.join(save_path, filename),
        os.path.join(save_path, 'checkpoint_' + str(epoch)+'.pth.tar'))
    print("[i] checkpoint saved in ", save_path)
    if is_best:
        shutil.copyfile(
            os.path.join(save_path, filename),
            os.path.join(save_path, 'model_best.pth.tar'))
    if epoch > 3:
        prev_checkpoint_filename = os.path.join(
            save_path, 'checkpoint_' + str(epoch - 3) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def save_image(img, fname):
    """
    :param img: image (numpy array, H x W x 3)
    :param fname: file name (string)
    """
    img = np.array(img).astype('uint8')
    if img.ndim == 3 and img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))
    im = Image.fromarray(img)
    im.save(fname)

def accuracy(qry_feat, ref_feat, qry_label, topk=[1,5,10]):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    N = qry_feat.shape[0]
    # M = ref_feat.shape[0]
    # topk.append(M//100)
    results = np.zeros([len(topk)])
    # for CVUSA, CVACT
    if N < 5000:
        qry_feat_norm = np.sqrt(np.sum(qry_feat**2, axis=1, keepdims=True))
        ref_feat_norm = np.sqrt(np.sum(ref_feat ** 2, axis=1, keepdims=True))
        similarity = np.matmul(qry_feat/qry_feat_norm, (ref_feat/ref_feat_norm).transpose())

        for i in range(N):
            ranking = np.sum((similarity[i,:]>similarity[i,qry_label[i]])*1.)
            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.
                    # print(ranking, k)
                    # ww = similarity[i,:]>=similarity[i,qry_label[i]]
                    # print(list(ww).index(True))
    else:
        # split the queries if the matrix is too large, e.g. VIGOR
        assert N % 4 == 0
        N_4 = N // 4
        for split in range(4):
            qry_feat_i = qry_feat[(split*N_4):((split+1)*N_4), :]
            qry_label_i = qry_label[(split*N_4):((split+1)*N_4)]
            qry_feat_norm = np.sqrt(np.sum(qry_feat_i ** 2, axis=1, keepdims=True))
            ref_feat_norm = np.sqrt(np.sum(ref_feat ** 2, axis=1, keepdims=True))
            similarity = np.matmul(qry_feat_i / qry_feat_norm,
                                   (ref_feat / ref_feat_norm).transpose())
            for i in range(qry_feat_i.shape[0]):
                ranking = np.sum((similarity[i, :] > similarity[i, qry_label_i[i]])*1.)
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.        
    results = results/ qry_feat.shape[0] * 100.
    # print('Percentage-top1:{:.2f}, top5:{:.2f}, top10:{:.2f}, top1%:{:.2f}'.format(results[0], results[1], results[2], results[-1]))
    return results[:2]

