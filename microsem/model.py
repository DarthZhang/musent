import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

class WordModule(SerializableModule):
    def __init__(self, word_model, **kwargs):
        super().__init__()
        self.word_model = word_model

    def convert_dataset(self, dataset):
        dataset = np.stack(dataset)
        model_in = dataset[:, 1].reshape(-1)
        model_out = dataset[:, 0].flatten().astype(np.int)
        model_out = torch.autograd.Variable(torch.from_numpy(model_out))
        model_in = self.preprocess(model_in)
        model_in = torch.autograd.Variable(model_in)
        model_out = model_out.cuda()
        model_in = model_in.cuda()
        return (model_in, model_out)

    def preprocess(self, sentences):
        return torch.from_numpy(np.array(self.word_model.lookup(sentences)))

class NanoSem(WordModule):
    def __init__(self, word_model, **kwargs):
        super().__init__(word_model, **kwargs)
        self.dataset = kwargs["dataset"]
        self.down_project = nn.Linear(word_model.dim, 16)
        self.output_binary = nn.Linear(16, 2)
        self.output_fine = nn.Linear(16, 5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.word_model(x).squeeze(1)
        words = [self.down_project(self.dropout(x.view(x.size(0), -1))) for x in x.split(1, 1)]
        words = torch.stack(words, 1)
        x = torch.max(words, 1)[0]
        if self.dataset == data.DatasetEnum.SST_BINARY:
            return self.output_binary(x)
        elif self.dataset == data.DatasetEnum.SST_FINE:
            return self.output_fine(x)

class MicroSem(WordModule):
    def __init__(self, word_model, **kwargs):
        super().__init__(word_model, **kwargs)
        self.dataset = kwargs["dataset"]
        self.down_project = nn.Linear(word_model.dim, 16)
        self.gru = nn.GRU(16, 16, 1, batch_first=True)
        self.output_binary = nn.Linear(16, 2)
        self.output_fine = nn.Linear(16, 5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.word_model(x).squeeze(1)
        words = [self.down_project(self.dropout(x.view(x.size(0), -1))) for x in x.split(1, 1)]
        words = torch.stack(words, 1)
        _, x = self.gru(words)
        x = x.permute(1, 0, 2).contiguous().squeeze(1)
        if self.dataset == data.DatasetEnum.SST_BINARY:
            return self.output_binary(x)
        elif self.dataset == data.DatasetEnum.SST_FINE:
            return self.output_fine(x)

class WordEmbeddingModel(SerializableModule):
    def __init__(self, id_dict, weights, unknown_vocab=[], static=True, padding_idx=0):
        super().__init__()
        vocab_size = len(id_dict) + len(unknown_vocab)
        self.lookup_table = id_dict
        last_id = max(id_dict.values())
        for word in unknown_vocab:
            last_id += 1
            self.lookup_table[word] = last_id
        self.dim = weights.shape[1]
        self.weights = np.concatenate((weights, np.random.rand(len(unknown_vocab), self.dim) - 0.5))
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, self.dim, padding_idx=padding_idx)
        self.embedding.weight.data.copy_(torch.from_numpy(self.weights))
        if static:
            self.embedding.weight.requires_grad = False

    @classmethod
    def make_random_model(cls, id_dict, unknown_vocab=[], dim=300):
        weights = np.random.rand(len(id_dict), dim) - 0.5
        return cls(id_dict, weights, unknown_vocab, static=False)

    def forward(self, x):
        return self.embedding(x)

    def lookup(self, sentences):
        raise NotImplementedError

class SSTWordEmbeddingModel(WordEmbeddingModel):
    def __init__(self, id_dict, weights, static=True, unknown_vocab=[]):
        super().__init__(id_dict, weights, unknown_vocab, static=static, padding_idx=16259)

    def lookup(self, sentences):
        indices_list = []
        max_len = 0
        for sentence in sentences:
            indices = []
            for word in data.sst_tokenize(sentence):
                try:
                    index = self.lookup_table[word]
                    indices.append(index)
                except KeyError:
                    continue
            indices_list.append(indices)
            if len(indices) > max_len:
                max_len = len(indices)
        for indices in indices_list:
            indices.extend([self.padding_idx] * (max_len - len(indices))) 
        return indices_list

def set_seed(seed=0, no_cuda=False):
    np.random.seed(seed)
    if not no_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
