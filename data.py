import os
from io import open
import torch


class Vocabulary(object):
    def __init__(self):
        self.type2index = {}
        self.idx2type = []

    def add_type(self, token):
        if token not in self.type2index:
            self.idx2type.append(token)
            self.type2index[token] = len(self.idx2type) - 1
        return self.type2index[token]

    def __len__(self):
        return len(self.idx2type)


class Corpus(object):
    def __init__(self, path):
        self.vocab = Vocabulary()
        self.train = self.tokenize(os.path.join(path, "wiki.train.tokens"))
        self.valid = self.tokenize(os.path.join(path, "wiki.valid.tokens"))
        self.test = self.tokenize(os.path.join(path, "wiki.test.tokens"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.vocab.add_type(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ["<eos>"]
                ids = []
                for word in words:
                    ids.append(self.vocab.type2index[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
