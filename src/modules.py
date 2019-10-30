#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # feat.width, feat.hieght
        bi = 2 if cfg.RNN.BI else 1
        self.alignment =  nn.Linear(cfg.ENC.HIDDEN_SIZE + cfg.DEC.HIDDEN_SIZE, 1)
    def forward(enc_hidden, dec_hidden_prev):
        att_probs = F.softmax(self.alignment(torch.stack([enc_hidden, dec_hidden_prev], 0)), 1)
        return torch.mm(att_probs, enc_hidden)

class CNNModel(nn.Module):
    def __init__():
        super(CNNModel, self).__init__()
        self.layer1 = nn.Conv2d(in_channels, out_channels=512, kernel_size=3,
                                stride=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.layer2 = nn.Conv2d(512, out_channels=512, kernel_size=3, stride=1,
                                padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.layer3 = nn.Conv2d(512, out_channels=256, kernel_size=3, stride=1,
                                padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0)
        self.layer4 = nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1,
                                padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer5 = nn.Conv2d(256, out_channels=128, kernel_size=3, stride=1,
                                padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.layer6 = nn.Conv2d(128, out_channels=64, kernel_size=3, stride=1,
                                padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=2)

    def forward(self, x):
        x = self.bn1(self.layer1(x))
        x = self.pool1(self.bn2(self.layer2(x)))
        x = self.pool2(self.layer3(x))
        x = self.bn3(self.layer4(x))
        x = self.pool3(self.layer5(x))
        x = self.pool4(self.layer6(x))

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.rnn = nn.LSTM(cfg.RNN.INPUT_SIZE,
                           cfg.RNN.HIDDEN_SIZE,
                           num_layers=cfg.RNN.NUM_LAYERS,
                           batch_first=True,
                           dropout=cfg.RNN.DROPOUT,
                           bidirectional=cfg.RNN.BI,
                           nonlinearity='tanh')

    def forward(self, x, hidden_state):
        out, (h, c)= self.rnn(x, hidden_state)
        context = self.attention_module(out)
        return torch.stack([context, h])

class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg
    def forward(self,)
