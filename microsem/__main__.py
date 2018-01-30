import argparse
import os
import random

from torch import utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
import torch.nn as nn

import data
import model

class RandomSearch(object):
    def __init__(self, params):
        self.params = params

    def __iter__(self):
        param_space = list(GridSearch(self.params))
        random.shuffle(param_space)
        for param in param_space:
            yield param

class GridSearch(object):
    def __init__(self, params):
        self.params = params
        self.param_lengths = [len(param) for param in self.params]
        self.indices = [1] * len(params)

    def _update(self, carry_idx):
        if carry_idx >= len(self.params):
            return True
        if self.indices[carry_idx] < self.param_lengths[carry_idx]:
            self.indices[carry_idx] += 1
            return False
        else:
            self.indices[carry_idx] = 1
            return False or self._update(carry_idx + 1)

    def __iter__(self):
        self.stop_next = False
        self.indices = [1] * len(self.params)
        return self

    def __next__(self):
        if self.stop_next:
            raise StopIteration
        result = [param[idx - 1] for param, idx in zip(self.params, self.indices)]
        self.indices[0] += 1
        if self.indices[0] == self.param_lengths[0] + 1:
            self.indices[0] = 1
            self.stop_next = self._update(1)
        return result

def train(**kwargs):
    mbatch_size = kwargs["mbatch_size"]
    n_epochs = kwargs["n_epochs"]
    restore = kwargs["restore"]
    verbose = not kwargs["quiet"]
    lr = kwargs["lr"]
    weight_decay = kwargs["weight_decay"]
    seed = kwargs["seed"]
    dataset_name = kwargs["dataset"]
    kwargs["dataset"] = data.DatasetEnum.lookup(dataset_name)

    if not kwargs["no_cuda"]:
        torch.cuda.set_device(kwargs["gpu_number"])
    model.set_seed(seed)
    embed_loader = data.SSTEmbeddingLoader("data")

    id_dict, weights, unk_vocab_list = embed_loader.load_embed_data()
    word_model_static = model.SSTWordEmbeddingModel(id_dict, weights, unk_vocab_list)
    id_dict, weights, unk_vocab_list = embed_loader.load_embed_data()
    word_model_nonstatic = model.SSTWordEmbeddingModel(id_dict, weights, unknown_vocab=unk_vocab_list, static=False)
    if not kwargs["no_cuda"]:
        word_model_static.cuda()
        word_model_nonstatic.cuda()
    micro_sem = model.MicroSem(word_model_static, **kwargs)
    if restore:
        micro_sem.load(kwargs["input_file"])
    if not kwargs["no_cuda"]:
        micro_sem.cuda()

    micro_sem.train()
    criterion = nn.CrossEntropyLoss()
    output_layer = getattr(micro_sem, "output_{}".format(dataset_name))
    if kwargs["train_classifier_only"]:
        parameters = output_layer.parameters()
    else:
        parameters = list(filter(lambda p: p.requires_grad, micro_sem.parameters()))
    # optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    # optimizer = torch.optim.Adam(parameters, lr=5E-3, weight_decay=1E-4)
    optimizer = torch.optim.SGD(parameters, lr=0.005, momentum=0.9, weight_decay=1E-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=kwargs["dev_per_epoch"] * 41, mode="max")
    train_set, dev_set, test_set = data.SSTDataset.load_sst_sets("data", dataset=dataset_name)

    collate_fn = micro_sem.convert_dataset
    # sampler = data.BinningSampler(train_set.sentences, mbatch_size=mbatch_size)
    train_loader = utils.data.DataLoader(train_set, shuffle=True, batch_size=mbatch_size, drop_last=True, 
        collate_fn=collate_fn)
    dev_loader = utils.data.DataLoader(dev_set, batch_size=len(dev_set), collate_fn=collate_fn)
    test_loader = utils.data.DataLoader(test_set, batch_size=len(test_set), collate_fn=collate_fn)

    def evaluate(loader, dev=True):
        micro_sem.eval()
        tot_correct = 0
        tot_length = 0
        for m_in, m_out in loader:
            scores = micro_sem(m_in)
            loss = criterion(scores, m_out).cpu().data[0]
            n_correct = (torch.max(scores, 1)[1].view(m_in.size(0)).data == m_out.data).sum()
            # n_correct = (torch.round(scores).view(m_in.size(0)).data == m_out.data).sum()
            tot_correct += n_correct
            tot_length += m_in.size(0)
        accuracy = tot_correct / tot_length
        scheduler.step(accuracy)

        if dev and accuracy >= evaluate.best_dev:
            evaluate.best_dev = accuracy
            print("Saving best model ({})...".format(accuracy))
            micro_sem.save(kwargs["output_file"])
        if verbose:
            print("{} set accuracy: {}, loss: {}".format("dev" if dev else "test", accuracy, loss))
        micro_sem.train()
    evaluate.best_dev = 0

    for epoch in range(n_epochs):
        print("Epoch number: {}".format(epoch), end="\r")
        if verbose:
            print()
        i = 0
        for j, (train_in, train_out) in enumerate(train_loader):
            optimizer.zero_grad()

            if not kwargs["no_cuda"]:
                train_in.cuda()
                train_out.cuda()

            scores = micro_sem(train_in)
            loss = criterion(scores, train_out)
            loss.backward()
            optimizer.step()
            accuracy = (torch.max(scores, 1)[1].view(-1).data == train_out.data).sum() / mbatch_size
            i += mbatch_size
            if i % (len(train_set) // kwargs["dev_per_epoch"]) < mbatch_size:
                evaluate(dev_loader)
    evaluate(test_loader, dev=False)
    return evaluate.best_dev

def do_random_search(given_params):
    test_grid = [[0.15, 0.2], [4, 5, 6], [150, 200], [3, 4, 5], [200, 300], [200, 250]]
    max_params = None
    max_acc = 0.
    for args in RandomSearch(test_grid):
        sf, gc, hid, seed, fc_size, fmaps = args
        print("Testing {}".format(args))
        given_params.update(dict(n_epochs=7, quiet=True, gradient_clip=gc, hidden_Size=hid, seed=seed, 
            n_feature_maps=fmaps, fc_size=fc_size))
        dev_acc = train(**given_params)
        print("Dev accuracy: {}".format(dev_acc))
        if dev_acc > max_acc:
            print("Found current max")
            max_acc = dev_acc
            max_params = args
    print("Best params: {}".format(max_params))

def do_train(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="binary", type=str, choices=["fine", "binary"])
    parser.add_argument("--dev_per_epoch", default=16, type=int)
    parser.add_argument("--fc_size", default=200, type=int)
    parser.add_argument("--gpu_number", default=0, type=int)
    parser.add_argument("--hidden_size", default=150, type=int)
    parser.add_argument("--input_file", default="saves/model.pt", type=str)
    parser.add_argument("--lr", default=1E-1, type=float)
    parser.add_argument("--mbatch_size", default=32, type=int)
    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--n_feature_maps", default=200, type=float)
    parser.add_argument("--n_labels", default=5, type=int)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--output_file", default="saves/model.pt", type=str)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--rnn_type", choices=["lstm", "gru"], default="lstm", type=str)
    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--train_classifier_only", action="store_true", default=False)
    parser.add_argument("--weight_decay", default=1E-3, type=float)
    args, _ = parser.parse_known_args()
    args = vars(args)
    args.update(kwargs)
    train(**args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", default="train", type=str, choices=["train", "test", "dump"])
    parser.add_argument("--random_search", action="store_true", default=False)
    args, _ = parser.parse_known_args()
    if args.random_search:
        do_random_search(vars(args))
        return
    do_train(**vars(args))

if __name__ == "__main__":
    main()

