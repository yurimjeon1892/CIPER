import torch
import numpy as np

import os
import wandb


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


def retr_accuracy(qry_feat, ref_feat, qry_label):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    N = qry_feat.shape[0]
    M = ref_feat.shape[0]

    topk = [1, 5, 10]
    topk.append(M // 100)
    results = np.zeros([len(topk)])
    # for CVUSA, CVACT
    if N < 20000:
        qry_feat_norm = np.sqrt(np.sum(qry_feat**2, axis=1, keepdims=True))
        ref_feat_norm = np.sqrt(np.sum(ref_feat**2, axis=1, keepdims=True))
        similarity = np.matmul(
            qry_feat / qry_feat_norm, (ref_feat / ref_feat_norm).transpose()
        )

        for i in range(N):
            ranking = np.sum((similarity[i, :] > similarity[i, qry_label[i]]) * 1.0)
            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.0
    else:
        # split the queries if the matrix is too large, e.g. VIGOR
        # assert N % 4 == 0
        print("[!] Is N % 4 == 0 ?: ", (N % 4 == 0))
        N_D = N // 4
        for split in range(4):
            qry_feat_i = qry_feat[(split * N_D) : ((split + 1) * N_D), :]
            qry_label_i = qry_label[(split * N_D) : ((split + 1) * N_D)]
            qry_feat_norm = np.sqrt(np.sum(qry_feat_i**2, axis=1, keepdims=True))
            ref_feat_norm = np.sqrt(np.sum(ref_feat**2, axis=1, keepdims=True))
            similarity = np.matmul(
                qry_feat_i / qry_feat_norm, (ref_feat / ref_feat_norm).transpose()
            )
            for i in range(qry_feat_i.shape[0]):
                ranking = np.sum(
                    (similarity[i, :] > similarity[i, qry_label_i[i]]) * 1.0
                )
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.0
    results = results / qry_feat.shape[0] * 100.0
    # print("Percentage-top1:{:.2f}, top5:{:.2f}, top10:{:.2f}, top1%:{:.2f}".format(results[0], results[1], results[2], results[-1]))
    return results


def retr_accuracy_eval(qry_feat, ref_feat, qry_label, fname):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    N = qry_feat.shape[0]
    M = ref_feat.shape[0]

    topk = [1, 5, 10]
    topk.append(M // 100)
    results = np.zeros([len(topk)])
    # for CVUSA, CVACT
    if N < 20000:
        qry_feat_norm = np.sqrt(np.sum(qry_feat**2, axis=1, keepdims=True))
        ref_feat_norm = np.sqrt(np.sum(ref_feat**2, axis=1, keepdims=True))
        similarity = np.matmul(
            qry_feat / qry_feat_norm, (ref_feat / ref_feat_norm).transpose()
        )

        for i in range(N):
            ranking = np.sum((similarity[i, :] > similarity[i, qry_label[i]]) * 1.0)
            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.0
    else:
        # split the queries if the matrix is too large, e.g. VIGOR
        # assert N % 4 == 0
        print("[!] Is N % 4 == 0 ?: ", (N % 4 == 0))
        N_D = N // 4
        for split in range(4):
            qry_feat_i = qry_feat[(split * N_D) : ((split + 1) * N_D), :]
            qry_label_i = qry_label[(split * N_D) : ((split + 1) * N_D)]
            qry_feat_norm = np.sqrt(np.sum(qry_feat_i**2, axis=1, keepdims=True))
            ref_feat_norm = np.sqrt(np.sum(ref_feat**2, axis=1, keepdims=True))
            similarity = np.matmul(
                qry_feat_i / qry_feat_norm, (ref_feat / ref_feat_norm).transpose()
            )
            for i in range(qry_feat_i.shape[0]):
                ranking = np.sum(
                    (similarity[i, :] > similarity[i, qry_label_i[i]]) * 1.0
                )
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.0
    results = results / qry_feat.shape[0] * 100.0

    f = open(fname, "a")
    f.write("**cross-view image retrieval\n")
    f.write("top1, top5, top10, top1%\n")
    d = "{:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(
        results[0], results[1], results[2], results[-1]
    )
    f.write(d)
    f.close()
    return


def pose_accuracy(preds, gts):
    preds = np.concatenate(preds, 0)
    pred_shifts, pred_oriens = preds[:, :2], np.rad2deg(preds[:, 2])

    gts = np.concatenate(gts, 0)
    gt_shifts, gt_oriens = gts[:, :2], np.rad2deg(gts[:, 2])

    distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))  # [N]
    angle_diff = np.remainder(np.abs(pred_oriens - gt_oriens), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]
    return distance, angle_diff


def pose_accuracy_eval(preds, gts, fname):
    preds = np.concatenate(preds, 0)
    pred_shifts, pred_oriens = preds[:, :2], np.rad2deg(preds[:, 2])

    gts = np.concatenate(gts, 0)
    gt_shifts, gt_oriens = gts[:, :2], np.rad2deg(gts[:, 2])

    distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))  # [N]
    angle_diff = np.remainder(np.abs(pred_oriens - gt_oriens), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]

    metrics = [1, 5]
    angles = [1, 5]

    init_dis = np.sqrt(np.sum((gt_shifts) ** 2, axis=1))
    init_angle = np.abs(gt_oriens)

    f = open(fname, "a")
    f.write("**cross-view pose estimation\n")
    f.write("init location and orientation\n")
    line = "{:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(
        np.mean(init_dis),
        np.median(init_dis),
        np.mean(init_angle),
        np.median(init_angle),
    )
    f.write(line)
    f.write("diff location and orientation\n")
    line = "{:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(
        np.mean(distance),
        np.median(distance),
        np.mean(angle_diff),
        np.median(angle_diff),
    )
    f.write(line)

    diff_shifts_init = np.abs(pred_shifts - gt_shifts)

    diff_shifts = []
    for i in range(diff_shifts_init.shape[0]):
        c, s = np.cos(np.deg2rad(-gt_oriens[i])), np.sin(np.deg2rad(-gt_oriens[i]))
        R = np.array([[c, -s], [s, c]])
        diff_shift = R @ diff_shifts_init[i]
        diff_shifts.append(diff_shift)
    diff_shifts = np.array(diff_shifts)

    f.write("lateral 1m, 5m\n")
    line = ""
    for idx in range(len(metrics)):
        pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
        line += "{:.3f}, ".format(pred)
    f.write(line + "\n")

    f.write("longitudinal 1m, 5m\n")
    line = ""
    for idx in range(len(metrics)):
        pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
        line += "{:.3f}, ".format(pred)
    f.write(line + "\n")

    f.write("orientation 1deg, 5deg\n")
    line = ""
    angle_acc = {}
    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        angle_acc[str(angles[idx])] = pred

        line += "{:.3f}, ".format(pred)
    f.write(line + "\n")
    f.close()

    return


def adjust_learning_rate(optimizer, epoch, args):
    import math

    """Decay the learning rate based on schedule"""
    lr = args["lr"]
    if args["cos"]:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args["epochs"]))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_state(model, optimizer, epoch, is_best):
    # os.makedirs(save_path, exist_ok=True)
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    # save_name = os.path.join(save_path, "epoch_" + str(epoch)+".pth")
    # torch.save(state_dict, save_name)
    # print("[i] checkpoint saved in ", save_name)

    # if is_best:
    #     torch.save(state_dict, os.path.join(save_path, "model_best.pth"))
    #     print("[i] best checkpoint saved in ", os.path.join(save_path, "model_best.pth"))
    # if epoch > 3:
    #     prev_checkpoint_filename = os.path.join(
    #         save_path, "epoch_" + str(epoch - 3) + ".pth")
    #     if os.path.exists(prev_checkpoint_filename):
    #         os.remove(prev_checkpoint_filename)
    if wandb.run is not None:
        save_name = os.path.join(wandb.run.dir, "epoch_" + str(epoch) + ".pth")
        torch.save(state_dict, save_name)
        # wandb.save(save_name)


def print_pigeon():
    print(
        r"""\
                    .-''-.
                    / ,    \
                .-'`(o)    ;
                '-==.       |
                    `._...-;-.
                    )--'''   `-.
                    /   .        `-.
                    /   /      `.    `-.
                    |   \    ;   \      `-._________
                    |    \    `.`.;          -------`.
                    \    `-.   \\\\          `---...|
                    `.     `-. ```\.--'._   `---...|
                        `-.....7`-.))\     `-._`-.. /
                        `._\ /   `-`         `-.,'
                            / /
                            /=(_
                        -./--' `
                    ,^-(_
                    ,--' `                   
    
    """
    )
    return
