import torch
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
import time
import torch.nn as nn
from torchvision import models
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import confusion_matrix

seed_val = 0

torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True

nEpochs = 20
pittPath = '/Pitt/'
batchSize = 8
cnn = 'densenet'

modelName = 'audio'+cnn+str(nEpochs)

def writeStats(names, listas):
  for i, lista in enumerate(listas):
    with open('stats/' + modelName + names[i] + '.txt', 'w') as f:
        for item in lista:
            f.write("%s\n" % item)

def readIdx():
  audioPaths = []
  labels = []
  categories = ['dem', 'control']
  for cat in categories:
    if cat == 'control':
      path = pittPath + 'controlCha.txt'
    elif cat == 'dem':
      path = pittPath + 'dementiaCha.txt'
    index = open(path, 'r')
    files = index.readlines()
    for file in files:
      if cat == 'control':
        labels.append(0)
      elif cat == 'dem':
        labels.append(1)
      audioPaths.append(file[:-5] + '.npy')
  return audioPaths, labels

class Dataset:
    def __init__(self, audioPaths, labels):
        self.audioPaths = audioPaths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        audioSpecs = np.load(self.audioPaths[item])
        return audioSpecs, self.labels[item]
            


def collate_fn(batch, device):

    batchSize = len(batch)

    audio = torch.FloatTensor(batchSize, 3, 128, 250).fill_(0).to(device)
    label = torch.FloatTensor(batchSize).fill_(0).to(device)

    for i, (audioSpec, labelD) in enumerate(batch):
        audio[i] = torch.tensor(audioSpec)
        label[i] = labelD

    return audio, label


def getDataloaders(device):
    
    audioPaths, labels = readIdx()

    dataset = Dataset(audioPaths, labels)

    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    trainDataset, validationDataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    collate_fn_ = partial(collate_fn, device=device)
    trainIterator = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn_)
    validationIterator = DataLoader(validationDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn_)
    return trainIterator, validationIterator

class audioModel(nn.Module):
    def __init__(self, dropout = 0.2):
        super(audioModel, self).__init__()
        if cnn == 'resnet':
            self.cnn = models.resnet101(pretrained=True)
            outFeatures = self.cnn.fc.out_features
        elif cnn == 'densenet':
            self.cnn = models.densenet201(pretrained=True)
            outFeatures = self.cnn.classifier.out_features
        elif cnn == 'mobilenet':
            self.cnn = models.mobilenet_v2()
            outFeatures = self.cnn.classifier[1].out_features

        self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(outFeatures, 1)
            )

    def forward(self, audio):
        x = self.cnn(audio)
        x = self.classifier(x)
        return x


def train(model, trainIterator, valiadtionIterator, optimizer, scheduler, lossfn):

    for epoch in range(nEpochs):
        start = time.time()
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_size = 0
        for i, (audio, labels) in enumerate(trainIterator):
            outputs = model(audio)
 
            optimizer.zero_grad()

            predictions = torch.round(torch.sigmoid(outputs))
            acc = torch.round((predictions == labels.unsqueeze(1)).sum().float())
            loss = lossfn(outputs, labels.unsqueeze(1))
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            scheduler.step()
            # Statistics
            epoch_accuracy += acc.item()
            epoch_loss += loss.item()
            epoch_size += labels.size(0)
            
        end = time.time()
        print('Train Epoch: ', epoch + 1, ' | in ', end - start, ' seconds')
        print('Train Loss: ', epoch_loss / len(trainIterator))
        print('Train Accuracy: ', epoch_accuracy / epoch_size * 100)

        start = time.time()
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_size = 0
        conf_matrix = [[0,0], [0,0]]
        for i, (audio, labels) in enumerate(valiadtionIterator):
           
            with torch.no_grad():
                outputs = model(audio)
            
            predictions = torch.round(torch.sigmoid(outputs))
            

            conf = confusion_matrix(y_true = labels.cpu(), y_pred = predictions.squeeze(0).cpu(), labels=[0, 1])
            

            for i in range(2):
                for j in range(2):
                    conf_matrix[i][j] += conf[i][j]

            acc = torch.round((predictions == labels.unsqueeze(1)).sum().float())

            loss = lossfn(outputs, labels.unsqueeze(1))

            # Statistics
            epoch_accuracy += acc.item()
            epoch_loss += loss.item()
            epoch_size += labels.size(0)
            
        end = time.time()
        print('Validation Epoch: ', epoch + 1, ' | in ', end - start, ' seconds')
        print('Validation Loss: ', epoch_loss / len(validationIterator))
        print('Validation Accuracy: ', epoch_accuracy / epoch_size * 100)

        print('Validation confusion_matrix: ', conf_matrix)


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainIterator, validationIterator = getDataloaders(device)

    model = audioModel()
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.78))

    
    total_steps = len(trainIterator) * nEpochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


    train(model, trainIterator, validationIterator, optimizer, scheduler, lossfn)
    torch.save(model, 'modelos/' + modelName)
