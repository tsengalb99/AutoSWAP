"""Trainer script for sequential active learning / random sampling experiments."""
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

from external.NEAR import train_lib
from external.NEAR import dsl_bball
from datasets import BBallSeqDataset
from lib.custom_ap import bball_ap_score
from lib import query_methods
from lib import train_LSTM

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
parser.add_argument('--no_retrain_wl', default=False, action='store_true')
parser.add_argument('--no_weight_loss', default=False, action='store_true')
parser.add_argument('--use_diversity', default=False, action='store_true')
parser.add_argument('--dset')
parser.add_argument('--best_map', default=False, action='store_true')
parser.add_argument('--sample_dsl', type=float, default=1)
parser.add_argument('--pl_config_path', default=None, required=False)
args = parser.parse_args()

# set seed
torch.manual_seed(args.seed)
random.seed(args.seed)

# enforce various requirements
assert args.al_mode in set([
    'random',
    'uncertainty',
    'dg_entropy',
    'dg_kl',
    'dg_consensus_kl',
    'dg_max_kl',
    'active_weasul_maxkl',
])
assert args.wl_source in set([None, 'near', 'student'])

if args.al_mode.startswith('dg'):
    assert args.num_wl > 0 and args.wl_source is not None

if args.wl_source is None:
    assert args.num_wl == 0
    assert not args.perf_wl
elif args.wl_source in set(['near', 'student']):
    assert args.num_wl > 0

assert args.dset in set(['bball'])

# everything else


def get_dset_dsl_ap(dset, idx=None, mode=None):
    if dset == 'bball':
        train_set = BBallSeqDataset('train')
        valid_set = BBallSeqDataset('val')
        test_set = BBallSeqDataset('test')
        return train_set, valid_set, test_set, dsl_bball, bball_ap_score


def KL_uniform(probs):
    # D_KL(probs || uniform)
    num_classes = len(probs)
    uniform = np.ones(num_classes) / num_classes
    return np.sum(probs * np.log(probs / uniform))


def train(
        train_set,
        valid_set,
        test_set,
        ap_score,
        synthesis_args=None,
        sizes=[500, 1000, 1500, 2000, 3000, 4000, 5000],  # bball
        num_wl=1,
        device=0):

    X = train_set.features
    Y = train_set.annotations
    vX = valid_set.features
    vY = valid_set.annotations
    tX = test_set.features
    tY = test_set.annotations

    Y = np.expand_dims(Y, 1)
    vY = np.expand_dims(vY, 1)
    tY = np.expand_dims(tY, 1)

    sizes = sorted(sizes)
    sizes = list(filter(lambda size: size <= len(X), sizes))
    # add dummy step to the end to run on full ds
    sizes.append(sizes[-1])

    sel_inds = list(range(len(X)))
    random.shuffle(sel_inds)
    sel_inds = sorted(sel_inds[0:sizes[0]])

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
                            output_type='atom',
                            device=device,
                            existing_progs=existing_progs,
                        )
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
                    ap = ap_score(vY[:, -1], weak_labels_valid[-1])
                    print('NEAR {} BEST AP {}'.format(i, ap))

            elif args.wl_source == 'student':
                for i in range(num_wl):
                    (wlt, wlv, wlts), bap = train_LSTM.train_LSTM_student(
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

        # sample next indices
        if args.al_mode == 'uncertainty':
            next_inds, best_ap = query_methods.uncertainty_sampling(
                X,
                Y,
                weak_labels_train,
                vX,
                vY,
                weak_labels_valid,
                tX,
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
                is_seq=True)
        elif args.al_mode == 'random':
            next_inds, best_ap = query_methods.random_sampling(X,
                                                               Y,
                                                               weak_labels_train,
                                                               vX,
                                                               vY,
                                                               weak_labels_valid,
                                                               tX,
                                                               tY,
                                                               weak_labels_test,
                                                               ap_score,
                                                               sel_inds,
                                                               num_inds,
                                                               perfWL=args.perf_wl,
                                                               device=device,
                                                               no_weight_loss=args.no_weight_loss,
                                                               best_map=args.best_map,
                                                               is_seq=True)

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
