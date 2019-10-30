#!/usr/bin/env python3
import pickle
import queue
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


####################################################################################################
# Data Prep
####################################################################################################
class SheakspearData:
    def __init__(self, path, save_file, max_length):
        self.path = path
        self.max_length = max_length
        self.save_file = save_file
        self.data = []
        self.vocab2idx = defaultdict()
        self.idx2vocab = defaultdict()
        self.pickle_save()

    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            self.data = f.read().split('\n')[:-1]

    def create_dict(self):
        counter = Counter()
        special_token = {"<SOS>":0, "<EOS>": 1, "<UNK>": 2, "<PAD>": 3}
        for key, val in special_token.items():
            self.vocab2idx[key] = val
            self.idx2vocab[val] = key

        for lines in self.data:
            counter.update(lines.split(" "))
        counter.most_common(100)

        for word in counter:
            self.idx2vocab[len(self.vocab2idx)] = word
            self.vocab2idx[word] = len(self.vocab2idx)

    def encode_pad_data(self):
        padded_data = self.pad_data()
        for lines in padded_data:
            lines = map(lambda word: word if word in self.vocab2idx.keys() else "<UNK>", lines)
            lines = map(lambda word: self.vocab2idx[word], lines)
        return padded_data


    def pad_data(self):
        pad_data = self.data.copy()
        for line in pad_data:
            line = line.split(' ')
            line = ['SOS'] + line + ['EOS']
            if len(line) < self.max_length:
                line += ['PAD'] * (self.max_length - len(line))
        return pad_data

    def pickle_save(self):
        self.load_data()
        self.create_dict()
        enc_data = self.encode_pad_data()
        with open(self.save_file, 'wb') as f:
            pickle.dump(enc_data, f)

####################################################################################################
# Pytorch Dataset
####################################################################################################
class ShakespearLoader(Dataset):
    def __init__(self, data, batch_size):
        super(ShakespearLoader, self).__init__()
        self.data = data
        self.batch_size = batch_size
    def item(self, idx):
        self.data = nn.pack_padded_sequence(self.data[idx])
    def __len__(self):
        return len(self.data)
####################################################################################################
# Model
####################################################################################################
class CharRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, out_size, num_embedding):
        super(CharRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_embedding = num_embedding
        self.embed = nn.Embedding(self.num_embedding , embedding_dim=self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.num_embedding)

    def forward(self, x, hidden_state):
        x = self.embed(x)
        out, (h, c) = self.rnn(x, hidden_state)
        return F.softmax(self.linear(out), dim = 2)

####################################################################################################
# Beam Node
####################################################################################################
class BeamNode:
    def __init__(self, logp, parent, token, token_id, depth):
        self.logp = logp
        self.parent = parent
        self.depth = depth
        self.token = token
        self.token_id = token_id

####################################################################################################
# Beam Search
####################################################################################################
def beam_search(model, beam_width, max_depth, idx2vocab):
    def backtrack(node):
        result = []
        while node is not None:
            result.append(node.token)
            node = node.parent
        return list(reversed(result))

    # start
    node = BeamNode(1, None, '<SOS>', 0, 0)
    beam = queue.Queue(beam_width*max_depth)
    result = []
    beam.put(node)
    model.eval()
    while not beam.empty():
        node = beam.get()
        if node.depth >= max_depth or node.token=='<EOS>':
            if node.depth >= max_depth:
                break
            if node.token=='<EOS>':
                result.append(backtrack(node))
                continue
        # batch_size, seq_size, token_id
        hidden_state = torch.randn((1, 1, 10))
        cell_state = torch.randn((1, 1, 10))
        model_inp = torch.tensor([[node.token_id]])
        probs = model(model_inp, (hidden_state, cell_state))
        values, idxs = torch.topk(probs, beam_width)
        values = values[0, 0]
        idxs = idxs[0, 0]
        for idx, value in zip(idxs, values):
            idx = idx.item()
            value = value.item()
            token = idx2vocab[idx]
            token_id = idx
            parent = node
            logp = value
            depth = node.depth+1
            node = BeamNode(logp, parent, token, token_id, depth)
            if not beam.full():
                beam.put(node)
            else:
                break
    print(result)
