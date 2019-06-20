#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import torch
import csv
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn import metrics
import re


# In[107]:


class DataReader(Dataset):
    def __init__(self, file_path, task):
        self.file_path = file_path
        self.task = task
        df = pd.read_csv(self.file_path, delimiter="\t")
        self.id = df.loc[:,"id"].tolist()
        self.tweet = df.loc[:,"tweet"].tolist()
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
#             print(self.tweet[idx])
            token = self.clean_tweet(self.tweet[idx])
            clean_token = " ".join(token)
#             print(clean_token)
#             print(self.tweet[idx])
            bert_token = self.tokenizer.tokenize(clean_token)
#             print("bert",bert_token)
            bert_token.insert(0,"[CLS]")
#             index_token = self.tokenizer.convert_tokens_to_ids(token)
#             print(index_token)
#             print(len(index_token))
            label = torch.tensor([self.label_dica[self.subtask_a[idx]]])
            label = self.label_dica[self.subtask_a[idx]]
            return self.id[idx], bert_token, label
        
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
    
    def clean_tweet(self,text):
        ''' Function that is applied to every to tweet in the dataset '''
        
        # =========== TEXT ===========
        # Replace @USER by <user>
        text = re.compile(r'@USER').sub(r'user',text)

        # Replace URL by <url>
        text = re.compile(r'URL').sub(r'<url>',text)

        # Remove numbers :
        text = re.compile(r'[0-9]+').sub(r' ',text)

        # Remove some special characters
        text = re.compile(r'([\xa0_\{\}\[\]¬•$,:;/@#|\^*%().~`”"“-])').sub(r' ',text) 

        # Space the special characters with white spaces
        text = re.compile(r'([$&+,:;=?@#|\'.^*()%!"’“-])').sub(r' \1 ',text)
        
        # Replace some special characters : 
        replace_dict = {r'&' : 'and' , 
                        r'\+' : 'plus'}
        for cha in replace_dict:
            text = re.compile(str(cha)).sub(str(replace_dict[cha]),text)
            
        # Handle Emoji : translate some and delete the others
        text = self.handle_emoji(text)
        
        # Word delengthening : 
        text = re.compile(r'(.)\1{3,}').sub(r'\1\1',text)

        # Cut the words with caps in them : 
        text = re.compile(r'([a-z]+|[A-Z]+|[A-Z][a-z]+)([A-Z][a-z]+)').sub(r'\1 \2',text)
        text = re.compile(r'([a-z]+|[A-Z]+|[A-Z][a-z]+)([A-Z][a-z]+)').sub(r'\1 \2',text)        
        # =========== TOKENS ===========
        # TOKENIZE 
        text = text.split(' ')

        # Remove white spaces tokens
        text = [text[i] for i in range(len(text)) if text[i] != ' ']

        # Remove empty tokens
        text = [text[i] for i in range(len(text)) if text[i] != '']

        # Remove repetition in tokens (!!! => !)
        text = [text[i] for i in range(len(text)) if text[i] != text[i-1]]

        #  Handle the ALL CAPS Tweets 
        ### if ratio of caps in the word > 75% add allcaps tag <allcaps>
        caps_r = np.mean([text[i].isupper() for i in range(len(text))])
        if caps_r > 0.6 : 
            text.append('<allcaps>')

        # Lower Case : 
        text = [text[i].lower() for i in range(len(text))]

        return text
    
    def handle_emoji(self,text):
        # Dictionnary of "important" emojis : 
        emoji_dict =  {'♥️': ' love ',
                       '❤️' : ' love ',
                       '❤' : ' love ',
                       '😘' : ' kisses ',
                      '😭' : ' cry ',
                      '💪' : ' strong ',
                      '🌍' : ' earth ',
                      '💰' : ' money ',
                      '👍' : ' ok ',
                       '👌' : ' ok ',
                      '😡' : ' angry ',
                      '🍆' : ' dick ',
                      '🤣' : ' haha ',
                      '😂' : ' haha ',
                      '🖕' : ' fuck you '}

        for cha in emoji_dict:
            text = re.compile(str(cha)).sub(str(emoji_dict[cha]),text)
        # Remove ALL emojis
#         text = emoji.get_emoji_regexp().sub(r' ',text) 
        text = re.compile("([\U0001f3fb-\U0001f3ff])").sub(r'',text) 
        text = re.compile("([\U00010000-\U0010ffff])").sub(r'',text) 
        text = re.compile("(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])").sub(r'',text)

        # Add Space between  the Emoji Expressions : 
        text = re.compile("([\U00010000-\U0010ffff])").sub(r' \1 ',text) 
        return text


# In[108]:


file = "/home/cayu/nlp_final/project2_data/olid-training-v1.0.tsv"
dataset_a = DataReader(file, "a")
dataset_a[0]


# In[109]:


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
#         print("fuck",self.tweet[idx])
        token = self.clean_tweet(self.tweet[idx])
#         token = self.tweet[idx]
        clean_token = " ".join(token)
        bert_token = self.tokenizer.tokenize(clean_token)
        bert_token.insert(0,"[CLS]")

        label = self.label_dica[self.label[idx]]
        
        return self.id[idx], token,label
        
    def __len__(self):
        return len(self.id)
    
    def clean_tweet(self,text):
        ''' Function that is applied to every to tweet in the dataset '''
        
        # =========== TEXT ===========
        # Replace @USER by <user>
        text = re.compile(r'@USER').sub(r'user',text)

        # Replace URL by <url>
        text = re.compile(r'URL').sub(r'<url>',text)

        # Remove numbers :
        text = re.compile(r'[0-9]+').sub(r' ',text)

        # Remove some special characters
        text = re.compile(r'([\xa0_\{\}\[\]¬•$,:;/@#|\^*%().~`”"“-])').sub(r' ',text) 

        # Space the special characters with white spaces
        text = re.compile(r'([$&+,:;=?@#|\'.^*()%!"’“-])').sub(r' \1 ',text)
        
        # Replace some special characters : 
        replace_dict = {r'&' : 'and' , 
                        r'\+' : 'plus'}
        for cha in replace_dict:
            text = re.compile(str(cha)).sub(str(replace_dict[cha]),text)
            
        # Handle Emoji : translate some and delete the others
        text = self.handle_emoji(text)
        
        # Word delengthening : 
        text = re.compile(r'(.)\1{3,}').sub(r'\1\1',text)

        # Cut the words with caps in them : 
        text = re.compile(r'([a-z]+|[A-Z]+|[A-Z][a-z]+)([A-Z][a-z]+)').sub(r'\1 \2',text)
        text = re.compile(r'([a-z]+|[A-Z]+|[A-Z][a-z]+)([A-Z][a-z]+)').sub(r'\1 \2',text)        
        # =========== TOKENS ===========
        # TOKENIZE 
        text = text.split(' ')

        # Remove white spaces tokens
        text = [text[i] for i in range(len(text)) if text[i] != ' ']

        # Remove empty tokens
        text = [text[i] for i in range(len(text)) if text[i] != '']

        # Remove repetition in tokens (!!! => !)
        text = [text[i] for i in range(len(text)) if text[i] != text[i-1]]

        #  Handle the ALL CAPS Tweets 
        ### if ratio of caps in the word > 75% add allcaps tag <allcaps>
        caps_r = np.mean([text[i].isupper() for i in range(len(text))])
        if caps_r > 0.6 : 
            text.append('<allcaps>')

        # Lower Case : 
        text = [text[i].lower() for i in range(len(text))]

        return text
    
    def handle_emoji(self,text):
        # Dictionnary of "important" emojis : 
        emoji_dict =  {'♥️': ' love ',
                       '❤️' : ' love ',
                       '❤' : ' love ',
                       '😘' : ' kisses ',
                      '😭' : ' cry ',
                      '💪' : ' strong ',
                      '🌍' : ' earth ',
                      '💰' : ' money ',
                      '👍' : ' ok ',
                       '👌' : ' ok ',
                      '😡' : ' angry ',
                      '🍆' : ' dick ',
                      '🤣' : ' haha ',
                      '😂' : ' haha ',
                      '🖕' : ' fuck you '}

        for cha in emoji_dict:
            text = re.compile(str(cha)).sub(str(emoji_dict[cha]),text)
        # Remove ALL emojis
#         text = emoji.get_emoji_regexp().sub(r' ',text) 
        text = re.compile("([\U0001f3fb-\U0001f3ff])").sub(r'',text) 
        text = re.compile("([\U00010000-\U0010ffff])").sub(r'',text) 
        text = re.compile("(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])").sub(r'',text)

        # Add Space between  the Emoji Expressions : 
        text = re.compile("([\U00010000-\U0010ffff])").sub(r' \1 ',text) 
        return text
    
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


# In[110]:


test_file = "/home/cayu/nlp_final/project2_data/testset-levela.tsv"
test_csv = "/home/cayu/nlp_final/project2_data/labels-levela.csv"
test_dataset = ValDataReader(test_file, test_csv)
test_dataset[0]


# In[115]:


train_dataloader = DataLoader(dataset_a, batch_size=4, shuffle=False,collate_fn=my_collate)
val_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False,collate_fn=my_collate)
# for ids,seq,lab in val_dataloader:
#     print(ids,seq,lab)



# In[116]:


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
    for step, (ids,tweet,label) in enumerate(tqdm(train_dataloader)):
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
        score = metrics.f1_score(labels, predict,average='macro')
        print("F1 score", score)
        print("Epoch {} validation loss {:.6f}".format(i,sum(valid_loss)/len(valid_loss)))
        print("Epoch train loss",train_loss)


# In[ ]:





# In[ ]:




