import pandas as pd
import numpy as np
import torch
import csv
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn import metrics


class DataReader(Dataset):
    def __init__(self, file_path, task):
        self.file_path = file_path
        self.task = task
        df = pd.read_csv(self.file_path, delimiter="\t")
        self.id = df.loc[:,"id"].tolist()
        self.tweet = df.loc[:,"tweet_exp"].tolist()
        self.subtask_a = df.loc[:,"subtask_a"].tolist()
        self.subtask_b = df.loc[:,"subtask_b"].tolist()
        self.subtask_c = df.loc[:,"subtask_c"].tolist()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.label_dica = dict(zip(set(self.subtask_a),range(len(self.subtask_a))))
        self.label_dicb = dict(zip(set(self.subtask_b),range(len(self.subtask_b))))
        self.label_dicc = dict(zip(set(self.subtask_c),range(len(self.subtask_c))))
    
    def __getitem__(self, idx):
#         data_dic = dict(zip(ids,data))
        if self.task == "a":
            
            token = self.tokenizer.tokenize(self.tweet[idx])
            token.insert(0,"[CLS]")
#             index_token = self.tokenizer.convert_tokens_to_ids(token)
#             print(index_token)
#             print(len(index_token))
            label = torch.tensor([self.label_dica[self.subtask_a[idx]]])
            label = self.label_dica[self.subtask_a[idx]]
            return self.id[idx], token,label
        
        elif self.task == "b":
            token = self.tokenizer.tokenize(self.tweet[idx])
            token.insert(0,"[CLS]")
#             print(token)
            label = torch.tensor([self.label_dicb[self.subtask_b[idx]]])
            
            return self.id[idx], token, label
        
        elif self.task == "c":
            token = self.tokenizer.tokenize(self.tweet[idx])
            token.insert(0,"[CLS]")
            label = torch.tensor([self.label_dicc[self.subtask_c[idx]]])
            return self.id[idx], token, label
        
    def __len__(self):
        return len(self.id)
#         return 3

def my_collate(batch):
    sorted_batch = sorted(batch, key=lambda x:len(x[1]), reverse=True)
    ids = [x[0] for x in sorted_batch]
    sequences = [x[1] for x in sorted_batch]
    max_len = max([len(x[1]) for x in sorted_batch])
    for seq in sequences:
        while len(seq) < max_len:
            seq.append("[PAD]")

    label = [x[2] for x in sorted_batch]
#     print("label",label)
    labels = torch.tensor([x[2] for x in sorted_batch])
#     print(labels)
    return ids, sequences, labels

class ValDataReader(Dataset):
    
    def __init__(self, file_path, label_path):
        self.file_path = file_path
        self.label_path = label_path
        file_df = pd.read_csv(self.file_path, delimiter="\t")
        label_df = pd.read_csv(self.label_path,sep=",",header=None)
#         print(label_df)
        self.id =file_df.loc[:,"id"].tolist()
        self.tweet = file_df.loc[:,"tweet"].tolist()
        self.label = label_df.loc[:,1].tolist()
      
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.label_dica = dict(zip(set(self.label),range(len(self.label))))
        
        print(self.label_dica)
        
    def __getitem__(self, idx):
#         data_dic = dict(zip(ids,data))
            
        token = self.tokenizer.tokenize(self.tweet[idx])
        token.insert(0,"[CLS]")

        label = self.label_dica[self.label[idx]]
        
        return self.id[idx], token,label
        
    def __len__(self):
        return len(self.id)
    
def my_collate(batch):
    sorted_batch = sorted(batch, key=lambda x:len(x[1]), reverse=True)
    ids = [x[0] for x in sorted_batch]
    sequences = [x[1] for x in sorted_batch]
    max_len = max([len(x[1]) for x in sorted_batch])
    for seq in sequences:
        while len(seq) < max_len:
            seq.append("[PAD]")

    label = [x[2] for x in sorted_batch]
#     print("label",label)
    labels = torch.tensor([x[2] for x in sorted_batch])
#     print(labels)
    return ids, sequences, labels


test_file = "/home/robot/nlp_project2/label/testset-levela.tsv"
test_csv = "/home/robot/nlp_project2/label/labels-levela.csv"




file = "/home/robot/nlp_project2/olid-training-clean.csv"
dataset_a = DataReader(file, "a")
# dataset_b = DataReader(file, "b")
# dataset_c = DataReader(file, "c")
test_dataset = ValDataReader(test_file, test_csv)

train_dataloader = DataLoader(dataset_a, batch_size=8, shuffle=False,collate_fn=my_collate)
val_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False,collate_fn=my_collate)
       
learning_rate = 1e-3
warmup_proportion = 0.1
max_seq_length = 128


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)
model.to("cuda")
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


epoch = 50
for i in tqdm(range(epoch)):
    model.train()
    train_loss = []
    for step, (ids,tweet,label) in enumerate(train_dataloader):
        sequence = []
        for seq in tweet:
    #         print(seq)
            token_index = tokenizer.convert_tokens_to_ids(seq)
            token_index = torch.tensor(token_index)
    #         print(token_index)
            sequence.append(token_index)
        sequence = torch.stack(sequence).cuda()
        loss = model(sequence, labels=label.cuda())
#         logit = model(sequence)
#         pre = torch.max(logit,1)[1].tolist()
#         print("pre",pre)
        
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
#         print("train_loss",train_loss)
#         print("Training_loss: {:.6f}".format(loss.item()))
    train_loss = sum(train_loss) / len(train_loss)
    print("Epoch train loss",train_loss)
    
    model.eval()
    with torch.no_grad():
        valid_loss = []
        predict = []
        labels = []
        for step, (ids, tweet, label) in enumerate(val_dataloader):
            sequence = []
            for seq in tweet:
                token_index = tokenizer.convert_tokens_to_ids(seq)
                token_index = torch.tensor(token_index)
                sequence.append(token_index)
                
            sequence = torch.stack(sequence).cuda()
            loss = model(sequence, labels=label.cuda())
            logit = model(sequence)
            pre = torch.max(logit,1)[1].tolist()
            predict += pre
            print("predict",predict)
            valid_loss.append(loss.item())
            label = label.tolist()
            labels += label
        score = metrics.f1_score(labels, predict)
        print("F1 score", score)
        print("Epoch {} validation loss {:.6f}".format(i,sum(valid_loss)/len(valid_loss)))
        print("Epoch train loss",train_loss)