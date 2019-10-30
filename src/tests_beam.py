#!/usr/bin/env python3

import torch
from utils.beam_decoder import CharRNN, beam_search, SheakspearData

def test_dataset():
    data = SheakspearData('./corpus.txt', save_file='./corpus.pkl', max_length=5)
    print(data.vocab2idx)
    print(data.idx2vocab)


def test_beam_search():
    data = SheakspearData('./corpus.txt', save_file='./corpus.pkl', max_length=5)
    idx2vocab = data.idx2vocab
    num_embedding = len(idx2vocab)
    model = CharRNN(embed_size=5, hidden_size=10, out_size=5, num_embedding=num_embedding)
    beam_search(model, beam_width=3, max_depth=20, idx2vocab=idx2vocab)

if __name__ == "__main__":
    test_beam_search()
