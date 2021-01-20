import json,os,copy
import numpy as np
# import keras
# from keras.preprocessing.text import text_to_word_sequence
import requests
from vncorenlp import VnCoreNLP
annotator = VnCoreNLP("./VNCORE/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m',port=9000)
print('call annotator')
import re,copy
with open('./DB/vocab.json','r+') as f:
    vocabs = json.load(f)
path_data  = './DB/'
import sys
if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans
def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).

    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: ``!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n``,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.

    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

class Token(object):
    def __init__(self,vocabs):
        self.vocabs = copy.deepcopy(vocabs)
    
    def get(self,words):
        if words in self.vocabs.keys():
            return self.vocabs[words][1]
#         print(words)
        return self.vocabs['unk'][1]
    
    
    def __call__(self,inputs):
        result = text_to_word_sequence(inputs)
        return [self.get(word) for word in result]
        
class DataModel(object):
    
    def __init__(self, batch_size = 32,shuffle = True,tokenizer=None):
        super().__init__()
        self.token = tokenizer
        
#         self.token.add_tokens(["hero","skill"])
        
        with open("{}/data_trainx.json".format(path_data),"r") as f:
            self.data = json.load(f)
        # self.data = self.data[-2000:]
        with open("{}/champ.json".format(path_data),"r") as f:
            self.champ = json.load(f)
        with open("{}/table.json".format(path_data),"r") as f:
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
        x = ' '  + x + ' '
        pattern = '\s+[A-Z]\s+'
        z = re.findall(pattern,x)
        if len(z):
            entities = {"skill":[i.strip() for i in z]}
        else : entities = {}
        postion = []
        for zzz in self.champ:
            i = zzz['name']
            if len(re.findall("\s+{}\s+".format(i),x)) !=0:
                if 'hero' not in entities:
                    entities['hero'] = []

                entities["hero"].append(re.findall("\s+{}\s+".format(i),x)[0].strip())
                postion.append(x.split().index(i))
                # break
        
        postion = list(set(postion))
        postion.sort()
        postion = [(i,x.split()[i]) for i in postion]
        entities['postion'] = postion 
        return entities
    
    def replace_text(self,text):
        t = re.sub("{hero}","hero",text)
        for i in self.champ:
            t = re.sub(f"{i['name']}"," hero ",t)
            t = re.sub(f"{i['name'].lower()}"," hero ",t)
        t = re.sub("kĩ năng","skill",t)
        t = re.sub("[QWER]{1}","skill",t)
        return text.lower()
    
    def replace_table(self,text):
        # print(self.table)
        for i in self.table.keys():
            text = re.sub(f"\s+{i}\s+"," "+self.table[i]+" ",text)
        for i in [('khôngchị','không chị'),('tướngnào','tướng nào')]:
            text = text.replace(i[0],i[1])
        return self.replace_text(text)

    def replace_text2(self,text_):
        text = copy.copy(text_)
        for i in self.table.keys():
            text = re.sub(f"\s+{i}\s+"," "+self.table[i]+" ",text)
        for i in [('khôngchị','không chị'),('tướngnào','tướng nào')]:
            text = text.replace(i[0],i[1])
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
#             print(text)
            X.append(self.token(self.tokenizer(text)))
            Y.append(self.intents.index(intent))
#         print(X)
        # a = keras.preprocessing.sequence.pad_sequences(X)
        # b =  keras.utils.to_categorical(Y,num_classes=10)
#         print(a)
#         print(b)
        return X,Y
#         mask =  x['attention_mask']a['',b
        
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)


data = DataModel(tokenizer= Token(vocabs))


def preprocessing_text_incoming(text):
    print(data.tokenizer(text[0]),' --- token ---')
    text_normalize = [data.token(data.tokenizer(i)) for i in text]
    print(text_normalize)
    return np.array(text_normalize)



model_name='text_cl'
host='tf_docker_tensorflow-serving_1'
port=8500
conf_thresold = 0.95
def predict_text(text):
    print(text)
    text_replace = data.replace_text2(text)
    print(text_replace,text)
    entities = data.find_entities(text_replace)

    inputs = preprocessing_text_incoming([text])
    print(inputs.tolist())
    # print(inputs)
    datax = json.dumps({
        'instances':[
            {'embedding_input':inputs.tolist()[0]}
        ]
        

    })
    headers = {"content-type": "application/json"}
    # print('post')
    path= 'http://{}:{}/v1/models/{}:predict'.format(host, port, model_name)
    # print(path)
    
    json_response = requests.post(
        path,
        data=datax,
        headers=headers
    )
    print(json_response.text)
    # print(path)
    if json_response.status_code == 200:
        y_pre = json.loads(json_response.text)['predictions']
        y_pred = np.argmax(y_pre, axis=-1)
        # print(y_pre,y_pred)
        conf = np.max(y_pre)
        conf_thresold_ = conf_thresold
        
        data_response = {'intent':data.intents[y_pred[0]] if conf > conf_thresold_ else 'fallback',
                        'conf':conf}
        ranking = {}
        for index,i in enumerate(data.intents[:10]):
            ranking[i] = y_pre[0][index]

        data_response['ranking'] = ranking
        data_response['entities'] = entities
        return data_response

    return None

