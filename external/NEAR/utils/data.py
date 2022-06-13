import random
import torch
import multiprocessing as mp
import numpy as np
from collections.abc import Iterable


def flatten_batch(batch):
    if not isinstance(batch[0], Iterable) or len(batch[0]) == 1:
        return batch
    new_batch = []
    for traj_list in batch:
        new_batch.extend(traj_list)
    return new_batch


def flatten_tensor(batch_out):
    return torch.cat(batch_out)


def pad_minibatch(minibatch, num_features=-1, pad_token=-1, return_max=False):
    padded_mb = torch.nn.utils.rnn.pad_sequence(minibatch,
                                                batch_first=True,
                                                padding_value=pad_token).float()
    batch_lengths = [len(sequence) for sequence in minibatch]
    if return_max:
        longest_seq = max(batch_lengths)
        return padded_mb, batch_lengths, longest_seq
    return padded_mb, batch_lengths


def unpad_minibatch(minibatch, lengths, listtoatom=False):
    if listtoatom:
        return [mbi[l - 1] for mbi, l in zip(minibatch, lengths)]
    else:
        return [mbi[:l] for mbi, l in zip(minibatch, lengths)]


def dataset_tolists(trajs, labels):
    traj_list = map(lambda x: list(x), list(trajs))
    labels_list = list(torch.LongTensor(labels))
    return list(zip(traj_list, labels_list))


def normalize_data(train_data, valid_data, test_data):
    """Normalize features wrt. mean and std of training data."""
    _, seq_len, input_dim = train_data.shape
    train_data_reshape = np.reshape(train_data, (-1, input_dim))
    test_data_reshape = np.reshape(test_data, (-1, input_dim))
    features_mean = np.mean(train_data_reshape, axis=0)
    features_std = np.std(train_data_reshape, axis=0)
    train_data_reshape = (train_data_reshape - features_mean) / features_std
    test_data_reshape = (test_data_reshape - features_mean) / features_std
    train_data = np.reshape(train_data_reshape, (-1, seq_len, input_dim))
    test_data = np.reshape(test_data_reshape, (-1, seq_len, input_dim))
    if valid_data is not None:
        valid_data_reshape = np.reshape(valid_data, (-1, input_dim))
        valid_data_reshape = (valid_data_reshape - features_mean) / features_std
        valid_data = np.reshape(valid_data_reshape, (-1, seq_len, input_dim))
    return train_data, valid_data, test_data


def create_minibatches(all_items, batch_size):
    num_items = len(all_items)
    batches = []

    def create_single_minibatch(idxseq):
        curr_batch = []
        for idx in idxseq:
            curr_batch.append((all_items[idx]))
        return curr_batch

    item_idxs = list(range(num_items))
    while len(item_idxs) > 0:
        if len(item_idxs) <= batch_size:
            batch = create_single_minibatch(item_idxs)
            batches.append(batch)
            item_idxs = []
        else:
            # get batch indices
            batchidxs = []
            while len(batchidxs) < batch_size:
                rando = random.randrange(len(item_idxs))
                index = item_idxs.pop(rando)
                batchidxs.append(index)
            batch = create_single_minibatch(batchidxs)
            batches.append(batch)
    return batches


def prepare_datasets(train_data,
                     valid_data,
                     test_data,
                     train_labels,
                     valid_labels,
                     test_labels,
                     normalize=True,
                     train_valid_split=0.7,
                     batch_size=32):
    if normalize:
        train_data, valid_data, test_data = normalize_data(train_data, valid_data, test_data)

    trainset = dataset_tolists(train_data, train_labels)
    testset = dataset_tolists(test_data, test_labels)

    if valid_data is not None and valid_labels is not None:
        validset = dataset_tolists(valid_data, valid_labels)
    # Split training for validation set if validation set is not provided.
    elif train_valid_split < 1.0:
        split = int(train_valid_split * len(train_data))
        validset = trainset[split:]
        trainset = trainset[:split]
    else:
        split = int(train_valid_split)
        validset = trainset[split:]
        trainset = trainset[:split]

    # Create minibatches for training
    batched_trainset = create_minibatches(trainset, batch_size)

    return batched_trainset, validset, testset
