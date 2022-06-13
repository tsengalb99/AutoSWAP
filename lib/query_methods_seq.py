import numpy as np
import random
from collections import defaultdict, Counter

# all functions must return samples requested and best AP


def clip_probs(probs, thresh=1e-5):
    return np.clip(probs, thresh, 1 - thresh)


def uncertainty_sampling(trainX,
                         trainY,
                         trainWL,
                         validX,
                         validY,
                         validWL,
                         testX,
                         testY,
                         testWL,
                         ap_score,
                         sel_inds,
                         num_inds,
                         sampleWL=True,
                         perfWL=True,
                         device='cpu',
                         no_weight_loss=False,
                         best_map=False):
    if trainWL is not None and validWL is not None and testWL is not None:
        trainWL = np.concatenate(trainWL, -1)
        validWL = np.concatenate(validWL, -1)
        testWL = np.concatenate(testWL, -1)

    trainY = np.squeeze(trainY)
    validY = np.squeeze(validY)
    testY = np.squeeze(testY)

    if trainWL is not None:
        if sampleWL and perfWL:
            NN, best_ap = train_NN.train_NN((trainX[sel_inds], trainWL[sel_inds], trainY[sel_inds]),
                                            (validX, validWL, validY), (testX, testWL, testY),
                                            ap_score,
                                            device,
                                            no_weight_loss=no_weight_loss,
                                            best_map=best_map)
        elif sampleWL and not perfWL:
            NN, _ = train_NN.train_NN((trainX[sel_inds], trainWL[sel_inds], trainY[sel_inds]),
                                      (validX, validWL, validY), (testX, testWL, testY),
                                      ap_score,
                                      device,
                                      no_weight_loss=no_weight_loss,
                                      best_map=best_map)
            _, best_ap = train_NN.train_NN((trainX[sel_inds], None, trainY[sel_inds]),
                                           (validX, None, validY), (testX, None, testY),
                                           ap_score,
                                           device,
                                           no_weight_loss=no_weight_loss,
                                           best_map=best_map)
        elif not sampleWL and perfWL:
            _, best_ap = train_NN.train_NN((trainX[sel_inds], trainWL[sel_inds], trainY[sel_inds]),
                                           (validX, validWL, validY), (testX, testWL, testY),
                                           ap_score,
                                           device,
                                           no_weight_loss=no_weight_loss,
                                           best_map=best_map)
            NN, _ = train_NN.train_NN((trainX[sel_inds], None, trainY[sel_inds]),
                                      (validX, None, validY), (testX, None, testY),
                                      ap_score,
                                      device,
                                      no_weight_loss=no_weight_loss,
                                      best_map=best_map)
        else:
            NN, best_ap = train_NN.train_NN((trainX[sel_inds], None, trainY[sel_inds]),
                                            (validX, None, validY), (testX, None, testY),
                                            ap_score,
                                            device,
                                            no_weight_loss=no_weight_loss,
                                            best_map=best_map)
    else:
        NN, best_ap = train_NN.train_NN((trainX[sel_inds], None, trainY[sel_inds]),
                                        (validX, None, validY), (testX, None, testY),
                                        ap_score,
                                        device,
                                        no_weight_loss=no_weight_loss,
                                        best_map=best_map)

    total_inds = set(range(len(trainX)))
    remain_inds = sorted(list(total_inds.difference(set(sel_inds))))

    if len(remain_inds) > 0:
        if trainWL is not None and sampleWL:
            probs = train_NN.get_probs(NN, trainX[remain_inds], trainWL[remain_inds], device)
        else:
            probs = train_NN.get_probs(NN, trainX[remain_inds], None, device)

        # calculate entropy
        probs = clip_probs(probs)
        entropy = -np.sum(probs * np.log(probs), -1)

        indices = [(entropy[_], remain_inds[_]) for _ in range(len(entropy))]
        indices = sorted(indices, reverse=True)
        indices = [_[1] for _ in indices][0:min(num_inds, len(indices))]
    else:
        indices = []

    return indices, best_ap


def random_sampling(trainX,
                    trainY,
                    trainWL,
                    validX,
                    validY,
                    validWL,
                    testX,
                    testY,
                    testWL,
                    ap_score,
                    sel_inds,
                    num_inds,
                    perfWL=True,
                    device='cpu',
                    no_weight_loss=False,
                    best_map=False):
    if trainWL is not None and validWL is not None and testWL is not None:
        trainWL = np.concatenate(trainWL, -1)
        validWL = np.concatenate(validWL, -1)
        testWL = np.concatenate(testWL, -1)

    trainY = np.squeeze(trainY)
    validY = np.squeeze(validY)
    testY = np.squeeze(testY)

    if trainWL is not None and perfWL:
        _, best_ap = train_NN.train_NN((trainX[sel_inds], trainWL[sel_inds], trainY[sel_inds]),
                                       (validX, validWL, validY), (testX, testWL, testY),
                                       ap_score,
                                       device,
                                       no_weight_loss=no_weight_loss,
                                       best_map=best_map)
    else:
        _, best_ap = train_NN.train_NN((trainX[sel_inds], None, trainY[sel_inds]),
                                       (validX, None, validY), (testX, None, testY),
                                       ap_score,
                                       device,
                                       no_weight_loss=no_weight_loss,
                                       best_map=best_map)

    total_inds = set(range(len(trainX)))
    remain_inds = sorted(list(total_inds.difference(set(sel_inds))))

    if len(remain_inds) > 0:
        random.shuffle(remain_inds)
        indices = remain_inds[0:min(num_inds, len(remain_inds))]
    else:
        indices = []

    return indices, best_ap


def disagreement_sampling(trainX,
                          trainY,
                          trainWL,
                          validX,
                          validY,
                          validWL,
                          testX,
                          testY,
                          testWL,
                          ap_score,
                          sel_inds,
                          num_inds,
                          perfWL=True,
                          mode='dg_entropy',
                          device='cpu',
                          no_weight_loss=False,
                          best_map=False):
    # only used when WL exist
    assert trainWL is not None and validWL is not None and testWL is not None

    trainWL_concat = np.concatenate(trainWL, 1)
    validWL_concat = np.concatenate(validWL, 1)
    testWL_concat = np.concatenate(testWL, 1)

    trainY = np.squeeze(trainY)
    validY = np.squeeze(validY)
    testY = np.squeeze(testY)

    if perfWL:
        NN, best_ap = train_NN.train_NN(
            (trainX[sel_inds], trainWL_concat[sel_inds], trainY[sel_inds]),
            (validX, validWL_concat, validY), (testX, testWL_concat, testY),
            ap_score,
            device,
            no_weight_loss=no_weight_loss,
            best_map=best_map)
    else:
        NN, best_ap = train_NN.train_NN((trainX[sel_inds], None, trainY[sel_inds]),
                                        (validX, None, validY), (testX, None, testY),
                                        ap_score,
                                        device,
                                        no_weight_loss=no_weight_loss,
                                        best_map=best_map)

    total_inds = set(range(len(trainX)))
    remain_inds = sorted(list(total_inds.difference(set(sel_inds))))

    NN_probs = train_NN.get_probs(NN, trainX[remain_inds], trainWL_concat[remain_inds], device)
    clipped_probs = clip_probs(NN_probs)

    if mode == 'dg_entropy':
        # weighted entropy
        NN_entropy = -np.sum(clipped_probs * np.log(clipped_probs), -1)
        dg_sum = (best_ap**0.5) * NN_entropy
        for i in range(len(trainWL)):
            wt_ap = ap_score(validY, validWL[i])
            P = clip_probs(trainWL[i][remain_inds])
            WL_entropy = -np.sum(P * np.log(P), -1)
            dg_sum += wt_ap * WL_entropy
    elif mode == 'dg_kl':
        #kl between NN and NEAR
        entropy_wt = 1 / 10
        NN_entropy = -np.sum(clipped_probs * np.log(clipped_probs), -1)
        print(best_ap * NN_entropy * entropy_wt)
        dg_sum = best_ap * NN_entropy * entropy_wt  # add entropy
        for i in range(len(trainWL)):
            wt_ap = ap_score(validY, validWL[i])
            clipped_weak = clip_probs(trainWL[i][remain_inds])
            #KL = np.sum(clipped_weak * np.log(clipped_weak / clipped_probs), 1)
            KL = np.sum(clipped_probs * np.log(clipped_probs / clipped_weak), 1)
            dg_sum += wt_ap * KL
        print(dg_sum)
    elif mode == 'dg_consensus_kl':
        # https://modal-python.readthedocs.io/en/latest/content/query_strategies/Disagreement-sampling.html#max-disagreement
        probs_sum = clipped_probs
        for i in range(len(trainWL)):
            P = clip_probs(trainWL[i][remain_inds])
            probs_sum += P
        probs_sum /= (len(trainWL) + 1)
        max_dist = np.sum(clipped_probs * np.log(clipped_probs / probs_sum))
        for i in range(len(trainWL)):
            P = clip_probs(trainWL[i][remain_inds])
            max_dist = np.maximum(max_dist, np.sum(P * np.log(P / probs_sum), 1))
        dg_sum = max_dist

    scores = list(zip(dg_sum, range(len(remain_inds))))
    scores = sorted(scores, reverse=True)
    indices = sorted([scores[_][1] for _ in range(min(num_inds, len(scores)))])

    return indices, best_ap


def active_weasul_maxkl(
    trainX,
    trainY,
    trainWL,
    validX,
    validY,
    validWL,
    testX,
    testY,
    testWL,
    ap_score,
    sel_inds,
    num_inds,
    perfWL=True,
    device='cpu',
    no_weight_loss=False,
    best_map=False,
):
    # only used when WL exist
    assert trainWL is not None and validWL is not None and testWL is not None

    trainWL_concat = np.concatenate(trainWL, 1)
    validWL_concat = np.concatenate(validWL, 1)
    testWL_concat = np.concatenate(testWL, 1)

    trainY = np.squeeze(trainY)
    validY = np.squeeze(validY)
    testY = np.squeeze(testY)

    num_classes = 0
    for _ in [trainY, validY, testY]:
        num_classes = max(num_classes, int(np.max(_.flatten())) + 1)
    print(f'Active Weasul MaxKL detected {num_classes} classes')

    if perfWL:
        NN, best_ap = train_NN.train_NN(
            (trainX[sel_inds], trainWL_concat[sel_inds], trainY[sel_inds]),
            (validX, validWL_concat, validY), (testX, testWL_concat, testY),
            ap_score,
            device,
            no_weight_loss=no_weight_loss,
            best_map=best_map)
    else:
        NN, best_ap = train_NN.train_NN((trainX[sel_inds], None, trainY[sel_inds]),
                                        (validX, None, validY), (testX, None, testY),
                                        ap_score,
                                        device,
                                        no_weight_loss=no_weight_loss,
                                        best_map=best_map)

    total_inds = set(range(len(trainX)))
    remain_inds = sorted(list(total_inds.difference(set(sel_inds))))

    # bin data points
    sel_wl = np.transpose(np.stack([np.argmax(_[sel_inds], 1) for _ in trainWL]))
    remain_wl = np.transpose(np.stack([np.argmax(_[remain_inds], 1) for _ in trainWL]))

    # calculate p(y | lambda) for ground truth
    bin_gt = defaultdict(lambda: [])
    for i in range(len(sel_inds)):
        bin_gt[tuple(sel_wl[i])].append(trainY[sel_inds[i]])

    bin_dist_gt = defaultdict(lambda: np.ones(num_classes) / num_classes)
    print('ground truth distribution')
    for key in bin_gt:
        counts = dict(Counter(bin_gt[key]))
        counts_dist = np.zeros(num_classes)
        for _ in counts:
            counts_dist[int(_)] += counts[_]
        counts_dist /= np.sum(counts_dist)
        bin_dist_gt[key] = clip_probs(counts_dist)
        print(key, bin_dist_gt[key])

    # calculate p(y | lambda) for predictions from weak labels
    NN_probs = train_NN.get_probs(NN, trainX[remain_inds], trainWL_concat[remain_inds], device)
    clipped_probs = clip_probs(NN_probs)

    bin_pred = defaultdict(lambda: [])
    bin_pred_inds = defaultdict(lambda: [])
    for i in range(len(remain_inds)):
        bin_pred[tuple(remain_wl[i])].append(clipped_probs[i])
        bin_pred_inds[tuple(remain_wl[i])].append(i)

    bin_dist_pred = defaultdict(lambda: np.ones(num_classes) / num_classes)
    print('predicted distribution')
    for key in bin_pred:
        bin_dist_pred[key] = clip_probs(np.sum(bin_pred[key], 0) / len(bin_pred[key]))
        print(key, bin_dist_pred[key])

    # kl divergence KL(pred || gt)
    KL = {}
    for key in bin_dist_pred:
        P_ = bin_dist_pred[key]
        Q_ = bin_dist_gt[key]
        KL[key] = np.sum(P_ * np.log(P_ / Q_))

    # must extract to ensure ordering doesn't change
    KL_keys = []
    KL_probs = []
    total_KL = sum(list(KL.values()))
    for key in KL:
        KL_keys.append(key)
        KL_probs.append(KL[key] / total_KL)

    KL_probs = sorted(list(zip(KL_probs, KL_keys)), reverse=True)
    missing = 0
    next_indices = set([])
    total_requested = min(len(remain_wl), num_inds)
    for (prob, key) in KL_probs:
        requested = int(round(prob * total_requested))
        if requested > len(bin_pred_inds[key]):
            next_indices.update(bin_pred_inds[key])
            missing += requested - len(bin_pred_inds[key])
            del bin_pred_inds[key]
        else:
            random.shuffle(bin_pred_inds[key])
            next_indices.update(bin_pred_inds[key][0:requested])
            bin_pred_inds[key] = bin_pred_inds[key][requested:]

    nonselected_inds = []
    # correct for any rounding errors
    missing = max(missing, total_requested - len(next_indices))
    for key in bin_pred_inds:
        nonselected_inds += bin_pred_inds[key]

    # uncertainty sample everything else
    nonselected_inds = sorted(nonselected_inds)
    nonselected_clipped_probs = clipped_probs[nonselected_inds]
    nonselected_entropy = -np.sum(nonselected_clipped_probs * np.log(nonselected_clipped_probs), -1)
    nonselected_inds = sorted(list(zip(nonselected_entropy, nonselected_inds)), reverse=True)
    for _ in range(missing):
        next_indices.add(nonselected_inds[_][1])

    next_indices = list(next_indices)[0:min(total_requested, len(next_indices))]
    assert len(next_indices) == total_requested

    return next_indices, best_ap
