import string

#imports taught by Andrew
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

nltk.download('punkt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train, test = iter() #when we have data we will import them here

# inputs are the message history, outputs are either spambot or not spambot
testInputs = []
testOutputs = []

trainInputs = []
trainOutputs = []

# put the stuff in arrays
for y, x in train:
    trainInputs.append(x)
    trainOutputs.append(y)
for y, x in test:
    testInputs.append(x)
    testOutputs.append(y)

trainSize = 50000  # bigger = slow but better model

# take the first <trainsize> of the data
testInputs = testInputs[:trainSize]
testOutputs = testOutputs[:trainSize]

trainInputs = trainInputs[:trainSize]
trainOutputs = trainOutputs[:trainSize]

# similar to the yelp reviews, we want the words in the message history to be tokenized (split up)
tokTrainInputs = []
tokTestInputs = []

# this uses the nltk to grab only the words and not anything else
for x in tqdm(trainInputs):
    x = x.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokTrainInputs.append(nltk.word_tokenize(x))
for x in tqdm(testInputs):
    x = x.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokTestInputs.append(nltk.word_tokenize(x))

# to convert the words to something the nn can work with, we convert each word to numbers
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

wordToIndex = tokenizer.vocab
indexToWord = {wordToIndex[word]: word for word in wordToIndex}  # Grab Uncased Vocabulary
padToken = tokenizer.pad_token_id
unknownToken = tokenizer.unk_token_id

# Encode all Values
encTrainInputs = []
encTestInputs = []
maximum = 5000  # how many words we want to look at per case (this will be slow but accurate, since its reading an essay every time)

for sent in tokTrainInputs:
    newSent = []
    for word in sent:
        word = word.lower()
        if word not in wordToIndex:
            newSent.append(unknownToken)
        else:
            newSent.append(wordToIndex[word])

    newSent = newSent[:maximum]
    # Pad up to max length
    paddedSeq = [padToken] * maximum
    paddedSeq[:len(newSent)] = newSent

    encTrainInputs.append(paddedSeq)

for sent in tokTestInputs:
    newSent = []
    for word in sent:
        word = word.lower()
        if word not in wordToIndex:
            newSent.append(unknownToken)
        else:
            newSent.append(wordToIndex[word])

    newSent = newSent[:maximum]
    # Pad up to max length
    paddedSeq = [padToken] * maximum
    paddedSeq[:len(newSent)] = newSent

    encTestInputs.append(paddedSeq)

arrTrainInputs = np.array(encTrainInputs)
arrTestInputs = np.array(encTestInputs)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokenized_x, y):
        self.X = tokenized_x
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        return np.array(x), np.array(y)


# Create DataLoaders
BATCH_SIZE = 64


def get_dataloader(numExamples, batchSize):
    testDataset = Dataset(arrTrainInputs[:numExamples], trainOutputs[:numExamples])
    trainDataset = Dataset(arrTrainInputs[:numExamples], trainOutputs[:numExamples])
    testDataloader = torch.utils.data.DataLoader(testDataset, batchSize, shuffle=True)
    trainDataloader = torch.utils.data.DataLoader(trainDataset, batchSize, shuffle=False)
    return trainDataloader, testDataloader


trainDataloader, testDataloader = get_dataloader(trainSize, 128)


# due to the similar task, the code is very similar
# it too is trying to find patterns in a bunch of words
# in this case, trying to find things that scam/spam bot accounts (which would not be actively monitored by humans, since there are usually several hundred run by 1 person) do that normal people usually don't
class MessageHistoryRNN(nn.Module):
    def __init__(self, vocLen, padIdx):
        super().__init__()
        self.padIdx = padIdx
        self.vocabLen = vocLen
        self.embeddingDim = 100
        self.hiddenDim = 20
        self.numLayers = 1

        self.embeddingLayer = nn.Embedding(self.vocabLen, self.embeddingDim, padding_idx=self.paddingIdx)
        self.RNN = nn.LSTM(self.embeddingDin, self.hiddenDim, self.numLayers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(self.hidden_dim * 2 * self.num_layers, 1)


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


# for now this is just a copy of andrew's code, will change later to tune it
class Trainer:
    '''
    Trainer class to Train a binary classifier
    '''

    def __init__(self, model):
        self.device = device
        self.model = model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)

        self.loss_function = nn.BCEWithLogitsLoss()

    def save(self):
        # saves a model with the trainers
        torch.save(self.model.state_dict(), './model.pth')

    def training_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(x)

        loss = self.loss_function(outputs, y)

        loss.backward()
        self.optimizer.step()

    def evaluation_step(self, x, y):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        loss = self.loss_function(output, y)

        output = torch.sigmoid(output)

        output = torch.round(output)

        accuracy = output == y
        tp = torch.sum(accuracy)
        all = accuracy.reshape(-1).shape[0]
        return loss, tp / all

    def train_model(self, train_dataloader):
        for x, y in tqdm(train_dataloader):
            x = x.long().to(self.device)
            y = y.float().to(self.device)
            self.training_step(x, y)

    def evaluate_model(self, eval_dataloader):

        sum_loss = 0
        sum_accuracy = 0
        count = 0
        for x, y in tqdm(eval_dataloader):
            x = x.long().to(self.device)
            y = y.float().to(self.device)
            loss, acc = self.evaluation_step(x, y)
            sum_loss = sum_loss + loss
            sum_accuracy = sum_accuracy + acc
            count += 1

        sum_loss = sum_loss / count
        sum_accuracy = sum_accuracy / count
        print(sum_loss, sum_accuracy)

    def train_whole_model(self, num_epochs, train_dataloader, eval_dataloader):
        for epoch in range(num_epochs):
            print("--------TRAINING---------")
            self.train_model(train_dataloader)
            print("--------EVALUATION-------")
            self.evaluate_model(eval_dataloader)