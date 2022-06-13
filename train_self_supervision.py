"""Trainer script for non-sequential weak supervision experiments."""
import argparse
import json
import contextlib
import numpy as np
import os
import pickle as pkl
import random
import sys

sys.path.append('external/NEAR')
import time
import torch
from snorkel.labeling.model.label_model import LabelModel
import zss

from external.NEAR import train_lib
from external.NEAR import dsl_fly, dsl_mouse_extended
from datasets import FlyV1FrameDataset, MouseFrameDatasetExtended
from lib.custom_ap import fly_ap_score, mouse_ap_score
from lib import train_NN, train_DT, train_NN_self_supervision
from lib import discrim_model

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, required=False, default=0)
parser.add_argument('--run_id', required=True)
parser.add_argument('--wl_source', required=False, default=None)
parser.add_argument('--num_wl', type=int, required=False, default=0)
parser.add_argument('--seed', type=int, default=0, required=False)
parser.add_argument('--no_weight_loss', default=False, action='store_true')
parser.add_argument('--dset')
parser.add_argument('--best_map', default=False, action='store_true')
parser.add_argument('--num_labeled', type=int, required=True)
parser.add_argument('--use_diversity', default=False, action='store_true')
# use_discrim_model is for testing purposes only
parser.add_argument('--use_discrim_model', default=False, action='store_true')
parser.add_argument('--dt_forest_size', type=int, default=3)
parser.add_argument('--pl_config_path', default=None, required=False)

args = parser.parse_args()

# set seed
torch.manual_seed(args.seed)
random.seed(args.seed)

assert args.wl_source in set([None, 'near', 'student', 'decision_tree'])
assert args.dset in set(['fly', 'mouse_extended'])


def get_dset_dsl_ap(dset, idx=None, mode=None):
    if dset == 'fly':
        train_set = FlyV1FrameDataset(mode, "train", idx, float('inf'), reduced_inds=True)
        valid_set = FlyV1FrameDataset(mode, "val", idx, float('inf'), reduced_inds=True)
        test_set = FlyV1FrameDataset(mode, "test", idx, float('inf'), reduced_inds=True)
        return train_set, valid_set, test_set, dsl_fly, fly_ap_score
    if dset == 'mouse_extended':
        train_set = MouseFrameDatasetExtended('train')
        valid_set = MouseFrameDatasetExtended('val')
        test_set = MouseFrameDatasetExtended('test')
        return train_set, valid_set, test_set, dsl_mouse_extended, mouse_ap_score


def KL_uniform(probs):
    # D_KL(probs || uniform)
    num_classes = len(probs)
    uniform = np.ones(num_classes) / num_classes
    return np.sum(probs * np.log(probs / uniform))


def get_best_threshold(probs, gt, ap_score, len_frac=1 / 2):
    comparr = []
    for i in range(len(probs)):
        comparr.append((KL_uniform(probs[i]), probs[i], gt[i]))

    comparr = sorted(comparr, key=lambda x: x[0])
    sorted_probs = np.stack([x[1] for x in comparr])
    sorted_gt = np.concatenate([x[2] for x in comparr])

    best_thresh = 0.0
    best_ap = 0.0
    for i in range(0, int(len(comparr) * len_frac), 500):
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            test_ap = ap_score(sorted_gt[i:], sorted_probs[i:])
        if test_ap > best_ap:
            best_ap = test_ap
            best_thresh = comparr[i][0]

    return best_thresh, best_ap


def get_abstain_lf(probs, threshold):
    outarr = []
    total_abstain = 0
    for prob in probs:
        if KL_uniform(prob) < threshold:
            outarr.append(-1)
            total_abstain += 1
        else:
            outarr.append(np.argmax(prob))
    return outarr, total_abstain


def get_top_n_inds(probs, n):
    comparr = []
    for i in range(len(probs)):
        comparr.append((KL_uniform(probs[i]), i))
    comparr = sorted(comparr, reverse=True)
    return sorted([comparr[i][1] for i in range(n)])


def train(train_set,
          valid_set,
          test_set,
          ap_score,
          synthesis_args=None,
          sizes=[1, 2, 3, 4, 5],
          num_wl=0,
          device=0):
    X = np.expand_dims(train_set.features, 1)
    Y = np.expand_dims(train_set.annotations, 1)
    vX = np.expand_dims(valid_set.features, 1)
    vY = np.expand_dims(valid_set.annotations, 1)
    tX = np.expand_dims(test_set.features, 1)
    tY = np.expand_dims(test_set.annotations, 1)

    sizes = [int(_ * args.num_labeled) for _ in sizes]
    sizes = sorted(sizes)

    avail_inds = list(range(len(X)))
    random.shuffle(avail_inds)

    labeled_inds = avail_inds[0:args.num_labeled]
    avail_inds = avail_inds[args.num_labeled:]

    _, counts = np.unique(Y[labeled_inds], return_counts=True)

    if not args.no_weight_loss:
        class_weights = np.sqrt(1 / counts)
        class_weights = torch.tensor(class_weights / sum(class_weights)).float().to(device)
    else:
        class_weights = torch.tensor(np.ones(len(counts)) / len(counts)).float().to(device)

    if args.wl_source is not None:
        if args.wl_source == 'near':
            assert 'DSL' in synthesis_args and 'CUSTOM_WT' in synthesis_args
            # prep data
            prepped_data = train_lib.get_ds_tuple(X[labeled_inds], Y[labeled_inds], vX, vY,
                                                  np.concatenate([X, vX, tX], 0))

            # get near WL
            weak_labels = []
            pl_config = json.load(open(args.pl_config_path))
            existing_progs = []
            while len(weak_labels) < num_wl:
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    start_time = time.time()
                    near_wl, near_prog_str, near_prog = train_lib.run_near(
                        prepped_data,
                        class_weights=class_weights,
                        config=pl_config,
                        return_raw=True,
                        dsl=synthesis_args['DSL'],
                        custom_edge_costs=synthesis_args['CUSTOM_WT'],
                        device=device,
                        existing_progs=existing_progs)
                print('NEAR TOOK {} SECONDS TO RUN'.format(time.time() - start_time))
                weak_labels.append(near_wl)
                existing_progs.append(near_prog)
                print('SYNTHESIZED_PROGRAM', near_prog_str)

            weak_labels = weak_labels[0:num_wl]

            weak_labels_train = []
            weak_labels_valid = []
            weak_labels_test = []

            for i in range(len(weak_labels)):
                wl = weak_labels[i]
                weak_labels_train.append(wl[:len(X)])
                weak_labels_valid.append(wl[len(X):len(X) + len(vX)])
                weak_labels_test.append(wl[len(X) + len(vX):])
                ap = ap_score(vY, weak_labels_valid[-1])
                print('NEAR {} BEST AP {}'.format(i, ap))

        elif args.wl_source == 'student':
            assert num_wl > 0
            weak_labels_train = []
            weak_labels_valid = []
            weak_labels_test = []

            for i in range(num_wl):
                (wlt, wlv, wlts), bap = train_NN.train_NN_student(
                    (X, np.squeeze(Y)), (vX, np.squeeze(vY)), (tX, np.squeeze(tY)),
                    ap_score,
                    labeled_inds,
                    True,
                    device=device,
                    no_weight_loss=args.no_weight_loss,
                    best_map=args.best_map)
                print('STUDENT {} BEST AP {}'.format(i, bap))
                weak_labels_train.append(wlt)
                weak_labels_valid.append(wlv)
                weak_labels_test.append(wlts)

        elif args.wl_source == 'decision_tree':
            assert num_wl > 0
            weak_labels_train = []
            weak_labels_valid = []
            weak_labels_test = []

            if args.use_diversity:
                assert args.dt_forest_size >= args.num_wl
                dt_forest_preds = []
                dt_forest_baps = []
                dt_forest_trees = []

                for i in range(args.dt_forest_size):
                    dt_preds, bap, zss_tree = train_DT.train_DT(
                        (X, np.squeeze(Y)),
                        (vX, np.squeeze(vY)),
                        (tX, np.squeeze(tY)),
                        ap_score,
                        labeled_inds,
                    )
                    dt_forest_preds.append(dt_preds)
                    dt_forest_baps.append(bap)
                    dt_forest_trees.append(zss_tree)

                selected_dt = set([np.argmax(dt_forest_baps)])
                while len(selected_dt) != args.num_wl:
                    candidates = []
                    for i in range(len(dt_forest_trees)):
                        if i in selected_dt:
                            continue
                        div_sum = 0
                        for selected_idx in selected_dt:
                            div_sum += zss.simple_distance(dt_forest_trees[selected_idx],
                                                           dt_forest_trees[i])
                        candidates.append((div_sum, i))
                    candidates = sorted(candidates, reverse=True)
                    selected_dt.add(candidates[0][1])

                for selected_idx in selected_dt:
                    print(f'DECISION_TREE {selected_idx} BEST AP {dt_forest_baps[selected_idx]}')
                    wlt, wlv, wlts = dt_forest_preds[selected_idx]
                    weak_labels_train.append(wlt)
                    weak_labels_valid.append(wlv)
                    weak_labels_test.append(wlts)

            else:
                for i in range(num_wl):
                    (wlt, wlv, wlts), bap, _ = train_DT.train_DT(
                        (X, np.squeeze(Y)),
                        (vX, np.squeeze(vY)),
                        (tX, np.squeeze(tY)),
                        ap_score,
                        labeled_inds,
                    )
                    print('DECISION TREE {} BEST AP {}'.format(i, bap))
                    weak_labels_train.append(wlt)
                    weak_labels_valid.append(wlv)
                    weak_labels_test.append(wlts)

        # train weak label generative model
        if args.use_discrim_model:
            generated_wl = discrim_model.train_discrim_model(X,
                                                             Y,
                                                             weak_labels_train,
                                                             vX,
                                                             vY,
                                                             weak_labels_valid,
                                                             tX,
                                                             tY,
                                                             weak_labels_test,
                                                             ap_score,
                                                             labeled_inds,
                                                             device=device,
                                                             no_weight_loss=args.no_weight_loss,
                                                             best_map=args.best_map,
                                                             is_seq=False)
        else:
            abstain_labels_train = []
            for i in range(len(weak_labels_valid)):
                thresh, _ = get_best_threshold(weak_labels_valid[i], vY, ap_score, 3 / 5)
                abstain_labels, num_abstain = get_abstain_lf(weak_labels_train[i], thresh)
                print(f'ABSTAIN WEAK LABEL {i} ABSTAINED FROM {num_abstain}')
                abstain_labels_train.append(abstain_labels)
            abstain_labels_train = np.stack(abstain_labels_train, 1)

            weak_label_model = LabelModel(cardinality=weak_labels_train[0].shape[-1], device=device)
            weak_label_model.fit(abstain_labels_train, class_balance=counts / sum(counts))
            generated_wl = weak_label_model.predict_proba(abstain_labels_train)
    else:
        generated_wl = np.zeros((len(Y), int(np.max(Y.flatten())) + 1))
        generated_wl[np.arange(len(generated_wl)), Y[:, 0].astype(int)] = 1

    print('WEAK LABEL TRAIN AP')

    ap_scores = []
    for size in sizes:
        labeled_X = X[labeled_inds]
        labeled_Y = Y[labeled_inds][:, 0]
        labeled_Y_onehot = np.zeros((labeled_Y.shape[0], generated_wl.shape[-1]))
        labeled_Y_onehot[np.arange(len(labeled_Y_onehot)), labeled_Y.astype(int)] = 1

        vY_onehot = np.zeros((vY.shape[0], generated_wl.shape[-1]))
        vY_onehot[np.arange(len(vY_onehot)), vY[:, 0].astype(int)] = 1

        tY_onehot = np.zeros((tY.shape[0], generated_wl.shape[-1]))
        tY_onehot[np.arange(len(tY_onehot)), tY[:, 0].astype(int)] = 1

        unlabeled_X = X[avail_inds[0:size]]
        unlabeled_Y = generated_wl[avail_inds[0:size]]

        total_downstream_X = np.concatenate([labeled_X, unlabeled_X], 0)
        total_downstream_Y = np.concatenate([labeled_Y_onehot, unlabeled_Y], 0)

        print('CLASS FRACTIONS')
        print(counts / sum(counts))
        print(np.sum(total_downstream_Y, 0) / np.sum(total_downstream_Y.flatten()))

        _, bap = train_NN_self_supervision.train_NN_self_supervision(
            (total_downstream_X, total_downstream_Y),
            (vX, vY_onehot),
            (tX, tY_onehot),
            ap_score,
            device,
            no_weight_loss=args.no_weight_loss,
            best_map=True,
            override_weights=class_weights,
        )

        print('BEST AP WITH {} LABELED, {} UNLABELED SAMPLES: {}'.format(
            args.num_labeled, size, bap))
        ap_scores.append((size / args.num_labeled, bap))

    return ap_scores


def run_and_log(index, logdir):
    # index and features aren't used for mouse
    train_set, valid_set, test_set, dsl, ap_score = get_dset_dsl_ap(args.dset, index, 'features')
    if args.wl_source == 'near':
        synthesis_args = {'DSL': dsl.DSL_DICT, 'CUSTOM_WT': dsl.CUSTOM_EDGE_COSTS}
    else:
        synthesis_args = None

    if logdir is not None:
        with open(os.path.join(logdir, '{}.args'.format(index)), 'w') as f:
            f.write(json.dumps(vars(args), indent=2))
        with open(os.path.join(logdir, '{}.log'.format(index)), 'w') as f:
            with contextlib.redirect_stdout(f):
                ap_scores = train(train_set,
                                  valid_set,
                                  test_set,
                                  ap_score,
                                  synthesis_args=synthesis_args,
                                  num_wl=args.num_wl)
        with open(os.path.join(logdir, '{}_ap.pkl'.format(index)), 'wb') as out_f:
            pkl.dump(ap_scores, out_f)
    else:
        train(train_set,
              valid_set,
              test_set,
              ap_score,
              synthesis_args=synthesis_args,
              num_wl=args.num_wl)


if __name__ == '__main__':
    if args.run_id == 'debug':
        logdir = None
    else:
        logdir = os.path.join('logs', '{}'.format(args.run_id))
        if not os.path.exists(logdir):
            try:
                os.mkdir(logdir)
            except:
                assert os.path.exists(logdir)
    run_and_log(args.index, logdir)
