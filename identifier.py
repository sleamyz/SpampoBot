import string

#again the imports for machine learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext
import transformers

import numpy as np
import pandas as pd
import nltk
from tqdm.notebook import tqdm

# NLTK
nltk.download('punkt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#whenever the bot runs the train code again (to update the model periodically) a new instance of this is made with updated stuff
class Identifier:
    def __init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        self.wordToIndex = self.tokenizer.vocab
        self.indexToWord = {self.wordToIndex[word]: word for word in self.wordToIndex}  # Grab Uncased Vocabulary
        self.padToken = self.tokenizer.pad_token_id
        self.unknownToken = self.tokenizer.unk_token_id

        self.model = MessageHistoryRNN(len(self.wordToIndex), len(self.wordToIndex)-1)
        self.device = device
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load("PATH"))
        self.model.eval()

class MessageHistoryRNN(nn.Module):
    def __init__(self, vocLen, padIdx):
        super().__init__()
        self.padIdx = padIdx
        self.vocabLen = vocLen
        self.embeddingDim = 100
        self.hiddenDim = 20
        self.numLayers = 1

        self.embeddingLayer = nn.Embedding(self.vocabLen, self.embeddingDim, padding_idx=self.padIdx)
        self.RNN = nn.LSTM(self.embeddingDim, self.hiddenDim, self.numLayers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(self.hiddenDim * 2 * self.numLayers, 1)

    def forward(self, x):
        embedding = self.embedding_layer(x)
        _, (hidden, cell) = self.RNN(embedding)

        # hidden - The final state we use
        # hidden: Tensor(Num layers, N, Dimensional)
        hidden = hidden.transpose(0, 1)
        B = hidden.shape[0]

        hidden = hidden.reshape(B, -1)
        # classifier head
        logits = self.classifier(hidden)
        logits = logits.reshape(-1, )
        return logits
