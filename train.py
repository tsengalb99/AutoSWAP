"""Trainer script for non-sequential active learning / random sampling experiments."""
import argparse
import json
import contextlib
import numpy as np
import os
import pickle as pkl
import random
import sys
import zss

sys.path.append('external/NEAR')
import time
import torch

from external.NEAR import train_lib
from external.NEAR import dsl_fly, dsl_mouse_extended
from datasets import FlyV1FrameDataset, MouseFrameDatasetExtended
from lib.custom_ap import fly_ap_score, mouse_ap_score
from lib import query_methods
from lib import train_NN, train_DT

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, required=False, default=0)
parser.add_argument('--run_id', required=True)
parser.add_argument('--al_mode', required=True)
parser.add_argument('--wl_source', required=False, default=None)
parser.add_argument('--num_wl', type=int, default=0, required=False)
parser.add_argument('--sample_wl', default=False, action='store_true')
parser.add_argument('--perf_wl', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=0, required=False)
parser.add_argument('--use_raw_wl', default=False, action='store_true')
parser.add_argument('--no_weight_loss', default=False, action='store_true')
parser.add_argument('--use_diversity', default=False, action='store_true')
parser.add_argument('--dset')
parser.add_argument('--best_map', default=False, action='store_true')
parser.add_argument('--dt_forest_size', type=int, default=3)
# use_treba is for testing purposes only
parser.add_argument('--use_treba', default=False, action='store_true')
parser.add_argument('--pl_config_path', default=None, required=False)
args = parser.parse_args()

# set seed
torch.manual_seed(args.seed)
random.seed(args.seed)

# enforce various requirements
assert args.al_mode in [
    'random',
    'uncertainty',
    'dg_entropy',
    'dg_kl',
    'dg_consensus_kl',
    'dg_max_kl',
    'active_weasul_maxkl',
]

assert args.wl_source in [None, 'near', 'student', 'decision_tree']

if args.al_mode.startswith('dg'):
    assert args.num_wl > 0 and args.wl_source is not None

if args.wl_source is None:
    assert args.num_wl == 0
    assert not args.perf_wl
elif args.wl_source in set(['near', 'student']):
    assert args.num_wl > 0

assert args.dset in set(['fly', 'mouse_extended'])


def get_dset_dsl_ap(dset, idx=None, mode=None):
    if dset == 'fly':
        train_set = FlyV1FrameDataset(mode, "train", idx, float('inf'), reduced_inds=True)
        valid_set = FlyV1FrameDataset(mode, "val", idx, float('inf'), reduced_inds=True)
        test_set = FlyV1FrameDataset(mode, "test", idx, float('inf'), reduced_inds=True)
        return train_set, valid_set, test_set, dsl_fly, fly_ap_score
    if dset == 'mouse_extended':
        train_set = MouseFrameDatasetExtended('train', use_treba=args.use_treba)
        valid_set = MouseFrameDatasetExtended('val', use_treba=args.use_treba)
        test_set = MouseFrameDatasetExtended('test', use_treba=args.use_treba)
        return train_set, valid_set, test_set, dsl_mouse_extended, mouse_ap_score


def KL_uniform(probs):
    # D_KL(probs || uniform)
    num_classes = len(probs)
    uniform = np.ones(num_classes) / num_classes
    return np.sum(probs * np.log(probs / uniform))


def train(train_set,
          valid_set,
          test_set,
          ap_score,
          synthesis_args=None,
          sizes=[1000, 2000, 3500, 5000, 7500, 12500, 25000, 50000],
          num_wl=1,
          device=0):
    X = np.expand_dims(train_set.features, 1)
    Y = np.expand_dims(train_set.annotations, 1)
    vX = np.expand_dims(valid_set.features, 1)
    vY = np.expand_dims(valid_set.annotations, 1)
    tX = np.expand_dims(test_set.features, 1)
    tY = np.expand_dims(test_set.annotations, 1)
    if args.use_treba:
        Xtreba = np.expand_dims(train_set.treba, 1)
        vXtreba = np.expand_dims(valid_set.treba, 1)
        tXtreba = np.expand_dims(test_set.treba, 1)

    sizes = sorted(sizes)
    # add dummy step to the end to run on full ds
    sizes.append(sizes[-1])

    sel_inds = list(range(len(X)))
    random.shuffle(sel_inds)
    sel_inds = sorted(sel_inds[0:sizes[0]])
    # sanity check for random seed

    weak_labels_train = None
    weak_labels_valid = None
    weak_labels_test = None

    ap_scores = []
    # kl to uniform, lower is better
    diversity = []
    for target_size in sizes[1:]:
        num_inds = target_size - len(sel_inds)
        # get weak labels
        _, counts = np.unique(Y[sel_inds], return_counts=True)
        diversity.append((len(sel_inds), KL_uniform(counts / sum(counts))))

        if num_wl > 0:
            weak_labels_train = []
            weak_labels_valid = []
            weak_labels_test = []

            if args.wl_source == 'near':
                assert 'DSL' in synthesis_args and 'CUSTOM_WT' in synthesis_args
                # prep data
                prepped_data = train_lib.get_ds_tuple(X[sel_inds], Y[sel_inds], vX, vY,
                                                      np.concatenate([X, vX, tX], 0))
                if not args.no_weight_loss:
                    class_weights = np.sqrt(1 / counts)
                    class_weights = torch.tensor(class_weights /
                                                 sum(class_weights)).float().to(device)
                else:
                    class_weights = torch.tensor(np.ones(len(counts)) /
                                                 len(counts)).float().to(device)

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
                            return_raw=args.use_raw_wl,
                            dsl=synthesis_args['DSL'],
                            custom_edge_costs=synthesis_args['CUSTOM_WT'],
                            device=device,
                            existing_progs=existing_progs)
                    print('NEAR TOOK {} SECONDS TO RUN'.format(time.time() - start_time))
                    weak_labels.append(near_wl)
                    if args.use_diversity:
                        existing_progs.append(near_prog)
                    print('SYNTHESIZED_PROGRAM', near_prog_str)

                for i in range(len(weak_labels)):
                    wl = weak_labels[i]
                    weak_labels_train.append(wl[:len(X)])
                    weak_labels_valid.append(wl[len(X):len(X) + len(vX)])
                    weak_labels_test.append(wl[len(X) + len(vX):])
                    ap = ap_score(vY, weak_labels_valid[-1])
                    print('NEAR {} BEST AP {}'.format(i, ap))

            elif args.wl_source == 'student':
                for i in range(num_wl):
                    (wlt, wlv, wlts), bap = train_NN.train_NN_student(
                        (X, np.squeeze(Y)), (vX, np.squeeze(vY)), (tX, np.squeeze(tY)),
                        ap_score,
                        sel_inds,
                        args.use_raw_wl,
                        device=device,
                        no_weight_loss=args.no_weight_loss,
                        best_map=args.best_map)
                    print('STUDENT {} BEST AP {}'.format(i, bap))
                    weak_labels_train.append(wlt)
                    weak_labels_valid.append(wlv)
                    weak_labels_test.append(wlts)

            elif args.wl_source == 'decision_tree':
                if args.use_diversity:
                    assert args.dt_forest_size >= args.num_wl
                    dt_forest_preds = []
                    dt_forest_baps = []
                    dt_forest_trees = []
                    current_tree = None
                    for i in range(args.dt_forest_size):
                        dt_preds, bap, zss_tree = train_DT.train_DT(
                            (X, np.squeeze(Y)),
                            (vX, np.squeeze(vY)),
                            (tX, np.squeeze(tY)),
                            ap_score,
                            sel_inds,
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
                        print(
                            f'DECISION_TREE {selected_idx} BEST AP {dt_forest_baps[selected_idx]}')
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
                            sel_inds,
                        )
                        print('DECISION TREE {} BEST AP {}'.format(i, bap))
                        weak_labels_train.append(wlt)
                        weak_labels_valid.append(wlv)
                        weak_labels_test.append(wlts)

        whichX = Xtreba if args.use_treba else X
        whichvX = vXtreba if args.use_treba else vX
        whichtX = tXtreba if args.use_treba else tX

        # sample next indices
        if args.al_mode == 'uncertainty':
            next_inds, best_ap = query_methods.uncertainty_sampling(
                whichX,
                Y,
                weak_labels_train,
                whichvX,
                vY,
                weak_labels_valid,
                whichtX,
                tY,
                weak_labels_test,
                ap_score,
                sel_inds,
                num_inds,
                sampleWL=args.sample_wl,
                perfWL=args.perf_wl,
                device=device,
                no_weight_loss=args.no_weight_loss,
                best_map=args.best_map,
            )
        elif args.al_mode == 'random':
            next_inds, best_ap = query_methods.random_sampling(
                whichX,
                Y,
                weak_labels_train,
                whichvX,
                vY,
                weak_labels_valid,
                whichtX,
                tY,
                weak_labels_test,
                ap_score,
                sel_inds,
                num_inds,
                perfWL=args.perf_wl,
                device=device,
                no_weight_loss=args.no_weight_loss,
                best_map=args.best_map,
            )

        print('BEST AP WITH {} SAMPLES: {}'.format(len(sel_inds), best_ap))
        ap_scores.append((len(sel_inds), best_ap))
        sel_inds = sorted(sel_inds + next_inds)

    return ap_scores, diversity


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
                ap_scores, diversity = train(train_set,
                                             valid_set,
                                             test_set,
                                             ap_score,
                                             synthesis_args=synthesis_args,
                                             num_wl=args.num_wl)
        with open(os.path.join(logdir, '{}_ap.pkl'.format(index)), 'wb') as out_f:
            pkl.dump(ap_scores, out_f)
        with open(os.path.join(logdir, '{}_div.pkl'.format(index)), 'wb') as out_f:
            pkl.dump(diversity, out_f)
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
