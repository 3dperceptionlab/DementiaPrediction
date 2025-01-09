import torch
from torch.utils.data import DataLoader
from functools import partial
import time
import torch.nn as nn
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import confusion_matrix


seed_val = 0

torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True

nEpochs = 20
pittPath = '/Pitt/'
batchSize = 16
maxSeqLen = 512
textModel = 1
modelName = 'Texto'+str(textModel)+str(nEpochs)

def readFile(filePath):

  file = open(filePath, 'r')
  lines = file.readlines()

  utt = lines[0][:-1]
  return utt

def readIdx():
  utts = []
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
      utt = readFile(file[:-4] + 'txt')
      utts.append(utt)
      if cat == 'control':
        labels.append(0)
      elif cat == 'dem':
        labels.append(1)
  return utts, labels

class Dataset:
    def __init__(self, text, attentions, labels):
        self.text = text
        self.attentions = attentions
        self.labels = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return self.text[item], self.attentions[item], self.labels[item]
            


def collate_fn(batch, padVal, device):

    batchSize = len(batch)

    text = torch.LongTensor(batchSize, maxSeqLen).fill_(padVal).to(device)
    attentions = torch.IntTensor(batchSize, maxSeqLen).fill_(padVal).to(device)
    label = torch.FloatTensor(batchSize).fill_(0).to(device)

    for i, (transcript, attentionsD, labelD) in enumerate(batch):
        text[i] = transcript.detach().clone()
        attentions[i] = attentionsD.detach().clone()
        label[i] = labelD

    return text, attentions, label


def getDataloaders(device, tokenizer):
    utterances, labels = readIdx()

    tokenized_inputs = []
    attention_masks = []
    for utt in utterances:
        token = tokenizer.encode_plus(utt,
                            add_special_tokens = True,
                            max_length = maxSeqLen,           # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',   # Return pytorch tensors.
                            truncation = True
                      )
        tokenized_inputs.append(token['input_ids'])
        attention_masks.append(token['attention_mask'])

    dataset = Dataset(tokenized_inputs, attention_masks, labels)

    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    trainDataset, validationDataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    collate_fn_ = partial(collate_fn, device=device, padVal=0)
    trainIterator = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn_)
    validationIterator = DataLoader(validationDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn_)
    return trainIterator, validationIterator


class text(nn.Module):
    def __init__(self, dModel = 1024, dropout = 0.2):
      super(text, self).__init__()
      self.textModel = BertModel.from_pretrained('bert-base-uncased')
      self.dModel = dModel
      if textModel == 2:
        self.lstm = nn.LSTM(768, dModel, 1, batch_first = True, bidirectional = True)
      self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(dModel if textModel == 2 else 768, 1),
              )

    def forward(self, text, attention):
      textOut = self.textModel(text, attention_mask=attention)[0]
      if textModel == 1:
        output = textOut[:,0,:]
      elif textModel == 2:
        packed_input = pack_padded_sequence(textOut, [maxSeqLen for _ in range(text.size(0))], batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        output = output[range(len(output)), [maxSeqLen - 1 for _ in range(text.size(0))] ,:self.dModel]

      output = self.classifier(output)
      return output

def train(model, trainIterator, valiadtionIterator, optimizer, scheduler, lossfn):

    for epoch in range(nEpochs):
        start = time.time()
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_size = 0
        for i, (text, attention, labels) in enumerate(trainIterator):
            outputs = model(text, attention)
 
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
        for i, (text, attention, labels) in enumerate(valiadtionIterator):
           
            with torch.no_grad():
                outputs = model(text, attention)
            
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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainIterator, validationIterator = getDataloaders(device, tokenizer)

    model = text()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.1)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.78))

    
    total_steps = len(trainIterator) * nEpochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


    train(model, trainIterator, validationIterator, optimizer, scheduler, lossfn)
    torch.save(model, 'modelos/' + modelName)
