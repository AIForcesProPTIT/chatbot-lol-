
import os
import requests
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import json
import glob,os,sys,copy
import tensorflow as tf
from tensorflow import keras

from transformers import RobertaModel,PhobertTokenizer, TFRobertaModel

# import torch
import re
from vncorenlp import VnCoreNLP
annotator = VnCoreNLP("./VNCORE/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m',port=9000)
print('load done annotator')


class DataModel(keras.utils.Sequence):
    
    def __init__(self, batch_size = 32,shuffle = True):
        super().__init__()
        self.token = PhobertTokenizer.from_pretrained("./DB/token/")
        
        # self.token.add_tokens(["hero","skill"])
        # self.token.save_pretrained("./DB/token")
        
        with open("./DB/data_train.json","r") as f:
            self.data = json.load(f)
        with open("./DB/champ.json","r") as f:
            self.champ = json.load(f)
        with open("./DB/table.json","r") as f:
            self.table = json.load(f)
        self.intents = [
            "build_item","support_socket","counter",
                    "be_countered","skill_up",'how_to_play',"combo",
                    "combine_with",'how_to_use_skill','introduce',
                    'previous_question',
                    'chalenger_bot'
        ]
        self.batch_size  = batch_size
        self.shuffle = shuffle
        
        self.on_epoch_end()
        np.random.shuffle(self.indexes)
    
    def find_entities(self,x,verbose = False):
        print('find ' ,x)
        x = ' '  + x + ' '
        pattern = '\s+[A-Z]\s+'
        z = re.findall(pattern,x)
        if len(z):
            entities = {"skill":z[0].strip()}
        else:entities = {}
        postion = []
        for zzz in self.champ:
            i = zzz['name']
            if len(re.findall("\s+{}\s+".format(i),x)) !=0:
                postion.append(x.split().index(i))
                # break 
        
        postion = list(set(postion))
        postion.sort()
        postion = [(i,x.split()[i]) for i in postion]
        if len(postion) :
            entities['hero'] = []
            for ix in postion:
                entities['hero'].append(ix[1])
        entities['postion'] = postion 
        return entities
                
        # return entities
    
    def replace_text(self,text):
        t = re.sub("hero","hero",text)
        for i in self.champ:
            t = re.sub(f"{i['name']}"," hero ",t)
        t = re.sub("kĩ năng","skill",t)
        t = re.sub("[QWER]{1}","skill",t)
        return text
    def replace_text2(self,text_):
        text =  " " + copy.copy(text_)+ " "
        for i in self.table.keys():
            text = re.sub(f"\s+{i}\s+"," "+self.table[i]+" ",text)
        for i in [('khôngchị','không chị'),('tướngnào','tướng nào')]:
            text = text.replace(i[0],i[1])
        # print(text)
        return text
    def replace_table(self,text_):
        text = " " + copy.copy(text_)+ " "
        for i in self.table.keys():
            # if i in text:
                # print(text,i)
            text = text.replace(i,self.table[i])
        for i in self.champ:
            name = i['name']
            if name in text:
                text = text.replace(name,' hero ')
        print(text)
        return text
    
    def tokenizer(self,text):
        t = self.replace_table(text)
        sentences =[]
        for sen in annotator.annotate(t)['sentences']:
            sentences = sentences + ["."] + sen
        sentences = sentences[1:]
        return " ".join([i['form'] for i in sentences])
    
    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y
    
    def __data_generation(self,indexes):
        X = []
        Y=  []
        
        for ind in indexes:
            text,intent = self.data[ind]
            X.append(self.tokenizer(text))
            Y.append(self.intents.index(intent))
#         print(X)
        a = self.token(X,return_tensors='tf', padding=True, truncation=True)
        b =      keras.utils.to_categorical(Y,num_classes=10)
        return (a['input_ids'],a['attention_mask']),b

        
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
# import pandas as pd
data2 = DataModel()
def preprocessing_text_incoming_model2(text):
    text = text.replace("."," ")
    text_normalize = [data2.tokenizer(text.lower())]
    inputs = data2.token(text_normalize,return_tensors='tf', padding=True, truncation=True)

    return inputs['input_ids'],inputs['attention_mask']


model_name='text_cl'
host='tensorflow-serving'
port=8500
conf_thresold = 0.80
def predict_text(text):
    # print(text)
    # text_replace = data2.replace_text2(text)
    # print(text_replace,text)
    entities = data2.find_entities(data2.replace_text2(text))

    inputs = preprocessing_text_incoming_model2(text)
    # print(inputs.tolist())
    # print(inputs)
    datax = json.dumps({
        'instances':[
            {
                'input_1':inputs[0].numpy().tolist()[0],
                'input_2':inputs[1].numpy().tolist()[0]
            },
            ]
        }
        )
    headers = {"content-type": "application/json"}
    # print('post')
    path= 'http://{}:{}/v1/models/{}:predict'.format(host, port, model_name)
    # print(path)
    
    json_response = requests.post(
        path,
        data=datax,
        headers=headers
    )
    # print(json_response.text)
    # print(path)
    if json_response.status_code == 200:
        y_pre = json.loads(json_response.text)['predictions']
        y_pred = np.argmax(y_pre, axis=-1)
        # print(y_pre,y_pred)
        conf = np.max(y_pre)
        conf_thresold_ = conf_thresold
        if data2.intents[y_pred[0]] in ['how_to_play']:conf_thresold_ = 0.94
        if data2.intents[y_pred[0]] in ['introduce']:conf_thresold_ = 0.95
        data_response = {'intent':data2.intents[y_pred[0]] if conf > conf_thresold_ else 'fallback',
                        'conf':conf}
        ranking = {}
        for index,i in enumerate(data2.intents[:10]):
            ranking[i] = y_pre[0][index]

        data_response['ranking'] = ranking
        data_response['entities'] = entities
        return data_response

    return None
import requests



if __name__ == "__main__":
    # data2.token.save_pretrained("./DB/token")
    preprocessing_text_incoming_model2('con này chơi như thế nào')


