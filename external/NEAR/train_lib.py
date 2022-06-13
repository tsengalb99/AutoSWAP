import argparse
import os
import pickle
import random
import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import numpy as np

# import program_learning stuff
from algorithms import *
from dsl_fly import DSL_DICT, CUSTOM_EDGE_COSTS
from program_graph import ProgramGraph
from utils.data import prepare_datasets
from utils.evaluation import label_correctness
from utils.logging import init_logging, print_program_dict, log_and_print, print_program
from utils.training import process_batch


def get_ds_tuple(x, y, vx, vy, pred_x, normalize=True, batch_size=32):
    assert len(x) == len(y)
    input_size = x.shape[-1]
    output_size = int(np.max(y.flatten()) + 1)
    print('Automatically detected input_size {} output_size {}'.format(input_size, output_size))
    yshape = 1 if len(y.shape) == 1 else y.shape[1]
    batched_trainset, validset, testset = prepare_datasets(
        x,
        vx,
        pred_x,
        y,
        vy,
        np.zeros([pred_x.shape[0], yshape]),  # dummy var
        normalize=normalize,
        batch_size=batch_size)
    return (batched_trainset, validset, testset, input_size, output_size)


def run_near(
    prepped_data,
    config,
    class_weights=None,
    return_raw=False,
    dsl=DSL_DICT,
    custom_edge_costs=CUSTOM_EDGE_COSTS,
    output_type="list",
    device='cuda:0',
    existing_progs=[],
):
    '''
    importable function for NEAR
    x & y is the training data, pred_x is the x values predictions are needed on
    config is the NEAR config
    '''
    batched_trainset, validset, testset, input_size, output_size = prepped_data

    config['num_labels'] = output_size
    config['lossfxn'] = nn.CrossEntropyLoss(weight=class_weights)

    if config['optimizer'] == 'adam':
        config['optimizer'] = optim.Adam
    elif config['optimizer'] == 'sgd':
        config['optimizer'] = optim.SGD

    if 'evalfxn' not in config:
        config['evalfxn'] = label_correctness

    # Initialize program graph
    input_type = "list"
    program_graph = ProgramGraph(dsl,
                                 custom_edge_costs,
                                 input_type,
                                 output_type,
                                 input_size,
                                 output_size,
                                 config['max_num_units'],
                                 config['min_num_units'],
                                 config['max_num_children'],
                                 config['max_depth'],
                                 config['penalty'],
                                 ite_beta=config['ite_beta'])

    algorithm = ASTAR_NEAR_DIVERSITY(
        frontier_capacity=config['frontier_capacity'],
        existing_progs=existing_progs,
    )

    # Run program learning algorithm
    best_program = algorithm.run(program_graph, batched_trainset, validset, config, device)[-1]
    program_str_best = print_program_dict(best_program)
    best_program = best_program['program']

    test_input, _ = map(list, zip(*testset))

    with torch.no_grad():
        predicted_vals = process_batch(best_program, test_input, output_type, output_size, device)
        if return_raw:
            predicted_vals = nn.Softmax()(predicted_vals.to('cpu')).numpy()
        else:
            predicted_vals = np.argmax(nn.Softmax()(predicted_vals.to('cpu')).numpy(), 1)
        return predicted_vals, program_str_best, best_program
