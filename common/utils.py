import torch
import numpy as np


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


def retr_accuracy(qry_feat, ref_feat, qry_label, topk=[1, 5, 10]):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    N = qry_feat.shape[0]
    M = ref_feat.shape[0]
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
        DENOM = 7
        print(
            "Watch out! Due to device issue, we devide the features with number: ",
            DENOM,
            ", if you evaluate for paper, you MUST check this!",
        )
        print("N: ", N, ", M: ", M, ", N_D: ", N // DENOM)
        # split the queries if the matrix is too large, e.g. VIGOR
        assert N % DENOM == 0  # ??
        N_D = N // DENOM
        for split in range(DENOM):
            print("split 1")
            qry_feat_i = qry_feat[(split * N_D) : ((split + 1) * N_D), :]
            print("split 2")
            qry_label_i = qry_label[(split * N_D) : ((split + 1) * N_D)]
            print("split 3")
            qry_feat_norm = np.sqrt(np.sum(qry_feat_i**2, axis=1, keepdims=True))
            print("split 4")
            ref_feat_norm = np.sqrt(np.sum(ref_feat**2, axis=1, keepdims=True))
            print("split 5")
            similarity = np.matmul(
                qry_feat_i / qry_feat_norm, (ref_feat / ref_feat_norm).transpose()
            )
            print("split 6")
            for i in range(qry_feat_i.shape[0]):
                ranking = np.sum(
                    (similarity[i, :] > similarity[i, qry_label_i[i]]) * 1.0
                )
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.0
            print("split 7")
    results = results / qry_feat.shape[0] * 100.0
    # print("Percentage-top1:{:.2f}, top5:{:.2f}, top10:{:.2f}, top1%:{:.2f}".format(results[0], results[1], results[2], results[-1]))
    return results


def pose_accuracy(results, targets):
    # shift_range_lons, shift_range_lats, rotation_ranges = [], [], []

    gts, preds = [], []
    for b in range(len(results)):
        arl_img_size = targets[b]["orig_size"].detach().cpu().numpy()
        meter_per_pixel = targets[b]["meter_per_pixel"][0].detach().cpu().numpy()

        tgt = targets[b]["boxes"][0].detach().cpu().numpy()
        tgt = np.array(
            [
                [
                    tgt[0] * arl_img_size[0] * meter_per_pixel,
                    tgt[1] * arl_img_size[1] * meter_per_pixel,
                    np.arctan2(tgt[3], tgt[2]),
                ]
            ]
        )
        gts.append(tgt)

        scores = results[b]["scores"].detach().cpu().numpy()
        shifts = results[b]["boxes"].detach().cpu().numpy()
        shifts_max = shifts[np.argmax(scores), :]
        shifts_max = np.array([[shifts_max[0], shifts_max[1], shifts_max[2]]])
        preds.append(shifts_max)

        # print("score: ", np.min(scores), np.max(scores))

    gts = np.concatenate(gts, 0)
    preds = np.concatenate(preds, 0)

    gt_shifts, gt_headings = gts[:, :2], np.rad2deg(gts[:, 2])
    pred_shifts, pred_headings = preds[:, :2], np.rad2deg(preds[:, 2])

    distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))  # [N]
    angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]

    init_dis = np.sqrt(np.sum(gt_shifts**2, axis=1))
    init_angle = np.abs(gt_headings)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    shift_acc = {}
    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100
        shift_acc[str(metrics[idx])] = pred

        line = (
            "distance within "
            + str(metrics[idx])
            + " meters (pred, init): "
            + str(pred)
            + " "
            + str(init)
        )
        # print(line)

    diff_shifts = np.abs(pred_shifts - gt_shifts)
    for idx in range(len(metrics)):
        pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 0]) < metrics[idx]) / init_dis.shape[0] * 100

        line = (
            "lateral      within "
            + str(metrics[idx])
            + " meters (pred, init): "
            + str(pred)
            + " "
            + str(init)
        )
        # print(line)
        # f.write(line + "\n")

        pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = (
            np.sum(np.abs(gt_shifts[:, 1]) < metrics[idx]) / diff_shifts.shape[0] * 100
        )

        line = (
            "longitudinal within "
            + str(metrics[idx])
            + " meters (pred, init): "
            + str(pred)
            + " "
            + str(init)
        )
        # print(line)
        # f.write(line + "\n")

    angle_acc = {}
    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        angle_acc[str(angles[idx])] = pred

        line = (
            "angle within "
            + str(angles[idx])
            + " degrees (pred, init): "
            + str(pred)
            + " "
            + str(init)
        )
        # print(line)

    for idx in range(len(angles)):
        pred = (
            np.sum((angle_diff < angles[idx]) & (diff_shifts[:, 0] < metrics[idx]))
            / angle_diff.shape[0]
            * 100
        )
        init = (
            np.sum(
                (init_angle < angles[idx]) & (np.abs(gt_shifts[:, 0]) < metrics[idx])
            )
            / angle_diff.shape[0]
            * 100
        )
        line = (
            "lat within "
            + str(metrics[idx])
            + " & angle within "
            + str(angles[idx])
            + " (pred, init): "
            + str(pred)
            + " "
            + str(init)
        )
        # print(line)

    # result = np.sum((distance < metrics[0]) & (angle_diff < angles[0])) / distance.shape[0] * 100

    # acc1 = np.sum((distance < metrics[0]) & (angle_diff < angles[0]))
    # acc5 = np.sum((distance < metrics[2]) & (angle_diff < angles[2]))

    return distance, angle_diff


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
