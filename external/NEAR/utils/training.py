import sys
import copy
import torch
import torch.nn as nn
import dsl

from torch.utils.data import Dataset, DataLoader
from utils.data import pad_minibatch, unpad_minibatch, flatten_tensor, flatten_batch
from utils.logging import log_and_print


def preprocess_dataset(batch, device):
    batch_input, batch_output = map(list, zip(*batch))
    pin_memory = device != 'cpu'
    # process input
    batch_input = [torch.tensor(traj, pin_memory=pin_memory) for traj in batch_input]
    batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
    # process output
    true_vals = torch.tensor(flatten_batch(batch_output), pin_memory=pin_memory).float()
    return (batch_padded.to(device,
                            non_blocking=True), batch_lens, true_vals.to(device, non_blocking=True))


def init_optimizer(program, optimizer, lr):
    queue = [program]
    all_params = []
    while len(queue) != 0:
        current_function = queue.pop()
        if issubclass(type(current_function), dsl.HeuristicNeuralFunction):
            current_function.init_model()
            all_params.append({'params': current_function.model.parameters(), 'lr': lr})
        elif current_function.has_params:
            current_function.init_params()
            all_params.append({'params': list(current_function.parameters.values()), 'lr': lr})
        else:
            for submodule, functionclass in current_function.submodules.items():
                queue.append(functionclass)
    curr_optim = optimizer(all_params, lr)
    return curr_optim


def process_batch(program, batch, output_type, output_size, device='cpu', process_inputs=True):
    if process_inputs:
        batch_input = [torch.tensor(traj, pin_memory=device != 'cpu') for traj in batch]
        batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
        batch_padded = batch_padded.to(device)
    else:
        batch_padded, batch_lens = batch

    out_padded = program.execute_on_batch(batch_padded, batch_lens)
    if output_type == "list":
        out_unpadded = unpad_minibatch(out_padded,
                                       batch_lens,
                                       listtoatom=(program.output_type == 'atom'))
    else:
        out_unpadded = out_padded
    if output_size == 1 or output_type == "list":
        return flatten_tensor(out_unpadded).squeeze()
    else:
        if isinstance(out_unpadded, list):
            out_unpadded = torch.cat(out_unpadded, dim=0).to(device)
        return out_unpadded


def prep_execute_and_train_dset(trainset, validset, device):
    trainset_processed = [preprocess_dataset(_, device) for _ in trainset]
    validset_processed = preprocess_dataset(validset, device)

    return trainset_processed, validset_processed  #(validation_input, validation_true_vals)


def execute_and_train(program,
                      prepped_validset,
                      prepped_trainset,
                      train_config,
                      output_type,
                      output_size,
                      neural=False,
                      device='cpu',
                      use_valid_score=False,
                      print_every=60):
    lr = train_config['lr']
    neural_epochs = train_config['neural_epochs']
    symbolic_epochs = train_config['symbolic_epochs']
    optimizer = train_config['optimizer']
    lossfxn = train_config['lossfxn']
    evalfxn = train_config['evalfxn']
    num_labels = train_config['num_labels']

    num_epochs = neural_epochs if neural else symbolic_epochs
    curr_optim = init_optimizer(program, optimizer, lr)

    #validation_input, validation_true_vals = prepped_validset
    #trainset_processed = prepped_trainset
    val_batch_padded, val_batch_lens, val_true_vals = prepped_validset
    if isinstance(lossfxn, nn.CrossEntropyLoss):
        val_true_vals = val_true_vals.long()

    best_program = None
    best_metric = float('inf')
    best_additional_params = {}

    for epoch in range(1, num_epochs + 1):
        for (batch_padded, batch_lens, true_vals) in prepped_trainset:
            predicted_vals = process_batch(program, (batch_padded, batch_lens), output_type,
                                           output_size, device, False)
            # TODO a little hacky, but easiest solution for now
            if isinstance(lossfxn, nn.CrossEntropyLoss):
                true_vals = true_vals.long()

            if predicted_vals.dim() == 1:
                predicted_vals = torch.unsqueeze(predicted_vals, 0)

            loss = lossfxn(predicted_vals, true_vals)
            curr_optim.zero_grad()
            loss.backward()
            curr_optim.step()

        # check score on validation set
        with torch.no_grad():
            predicted_vals = process_batch(program, (val_batch_padded, val_batch_lens), output_type,
                                           output_size, device, False)
            metric, additional_params = evalfxn(predicted_vals,
                                                val_true_vals,
                                                num_labels=num_labels)

        if use_valid_score:
            if metric < best_metric:
                best_program = copy.deepcopy(program)
                best_metric = metric
                best_additional_params = additional_params
        else:
            best_program = copy.deepcopy(program)
            best_metric = metric
            best_additional_params = additional_params

    # select model with best validation score
    program = copy.deepcopy(best_program)
    log_and_print("Validation score is: {:.4f}".format(best_metric))
    log_and_print("Average f1-score is: {:.4f}".format(1 - best_metric))
    log_and_print("Hamming accuracy is: {:.4f}".format(best_additional_params['hamming_accuracy']))

    return best_metric
