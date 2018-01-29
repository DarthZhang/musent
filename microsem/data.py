import os
import random
import re

import numpy as np
import torch.utils.data as data

def sst_tokenize(sentence):
    return sentence.split()

class SSTEmbeddingLoader(object):
    def __init__(self, dirname, dataset="fine", word2vec_file="word2vec.sst-1"):
        self.fmt = "stsa.%s.{}" % dataset
        self.dirname = dirname
        self.word2vec_file = word2vec_file

    def load_embed_data(self):
        weights = []
        id_dict = {}
        unk_vocab_set = set()
        with open(os.path.join(self.dirname, self.word2vec_file)) as f:
            for i, line in enumerate(f.readlines()):
                word, vec = line.replace("\n", "").split(" ", 1)
                vec = np.array([float(v) for v in vec.split(" ")])
                weights.append(vec)
                id_dict[word] = i
        for fmt in ("phrases.train", "dev", "test"):
            with open(os.path.join(self.dirname, self.fmt.format(fmt))) as f:
                for line in f.readlines():
                    for word in sst_tokenize(line):
                        if word not in id_dict and word not in unk_vocab_set:
                            unk_vocab_set.add(word)
        return (id_dict, np.array(weights), list(unk_vocab_set))

class BinningSampler(data.sampler.Sampler):
    def __init__(self, items, mbatch_size=64, cuts=[0, 3, 5, 7, 9, 15, np.inf]):
        self.items = items
        self.cut_indices = []
        for i in range(len(cuts) - 1):
            c1, c2 = cuts[i], cuts[i + 1]
            sentences_cut = []
            for j, item in enumerate(items):
                if c1 <= len(item[1].split()) < c2:
                    sentences_cut.append(j)
            self.cut_indices.append(sentences_cut)
        self.mbatch_size = mbatch_size

    def __iter__(self):
        cut_indices = self.cut_indices.copy()
        chunks = []
        for c in cut_indices:
            random.shuffle(c)
            for i in range(0, len(c) - self.mbatch_size, self.mbatch_size):
                chunks.append(c[i:i + self.mbatch_size])
        random.shuffle(chunks)
        cont_chunks = []
        for c in chunks:
            cont_chunks.extend(c)
        return iter(cont_chunks)

    def __len__(self):
        return len(self.items)

class SSTDataset(data.Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences
        # print(np.sum([int(cut1 <= len(s[1].split()) < cut) for s in self.sentences]))
        # print(np.sum([int(len(s[1].split()) >= cut) for s in self.sentences]))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    @classmethod
    def load_sst_sets(cls, dirname, dataset="fine"):
        fmt = "stsa.%s.{}" % dataset
        set_names = ["phrases.train", "dev", "test"]
        def read_set(name):
            data_set = []
            with open(os.path.join(dirname, fmt.format(name))) as f:
                for line in f.readlines():
                    sentiment, sentence = line.replace("\n", "").split(" ", 1)
                    data_set.append((sentiment, sentence))
            return np.array(data_set)
        return [cls(read_set(name)) for name in set_names]
