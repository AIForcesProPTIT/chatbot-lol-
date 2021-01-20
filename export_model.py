import tensorflow as tf
from tensorflow import keras
import numpy as np
import json,os,sys
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import text_to_word_sequence
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
with open('./word_embedding/vector.npy', 'rb') as f:
    vectors = np.load(f)

with open('./word_embedding/vocab.json','r+') as f:
    vocabs = json.load(f)

from vncorenlp import VnCoreNLP
annotator = VnCoreNLP("./web/VNCORE/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
import re,copy
from tensorflow.keras.layers import Dense,Lambda, dot, Activation, concatenate
path_data  = './web/DB'
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
        
class DataModel(keras.utils.Sequence):
    
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
            entities = {"skill":z[0].strip()}
        else : entities = {}
        for zzz in self.champ + [{'name':'Lee'},{"name":"Sin"},{'name':'Master'},{"name":"Mundo"},{"name":"Aurelion"},
                                {"name":"Sol"},{"name":"Dr."},{"name":"Miss"},
                                {"name":"Fortune"},{"name":"Twisted"},{"name":"Fate"},
                                {"name":"Xin"},{"name":"Zhao"},{"name":"Jarvan"},{"name":"IV"},
                                {"name":"Yi"}]:
            i = zzz['name']
            if len(re.findall("\s+{}\s+".format(i),x)) !=0:
                entities["hero"]=re.findall("\s+{}\s+".format(i),x)[0].strip()
                break
        if verbose:
            if 'hero' in entities.keys():pass
                
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
        for i in self.table.keys():
            if i in text:
                text = text.replace(i,self.table[i])
        for i in [('khôngchị','không chị'),('tướngnào','tướng nào')]:
            text = text.replace(i[0],i[1])
        return self.replace_text(text)
    
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
        a = tf.keras.preprocessing.sequence.pad_sequences(X)
        b =  keras.utils.to_categorical(Y,num_classes=10)
#         print(a)
#         print(b)
        return a,b
#         mask =  x['attention_mask']a['',b
        
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class Attention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, hidden_states):
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        """
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

def create_test_model(word_vectors):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(word_vectors.shape[0], word_vectors.shape[1], name="embedding", 
                                  embeddings_initializer=tf.keras.initializers.Constant(word_vectors), trainable=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True), name="bidi_lstm_1"),
        tf.keras.layers.Dropout(0.42, name="dropout_1"),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512//2,return_sequences=True), name="bidi_lstm_2"),
        tf.keras.layers.Dropout(0.42, name="dropout_2"),
        Attention(),
        tf.keras.layers.Dense(256, activation='tanh', kernel_regularizer=tf.keras.regularizers.l1_l2(), name="dense_1"),
#         tf.keras.layers.Dropout(0.42, name="dropout_3"),
        tf.keras.layers.Dense(256, activation='tanh',kernel_regularizer=tf.keras.regularizers.l1_l2(), name="dense_2"),
        tf.keras.layers.Dropout(0.42, name="dropout_4"),
        tf.keras.layers.Dense(64, activation='tanh', name="dense_3"),
        tf.keras.layers.Dense(10, activation="softmax", name="dense_output")
    ])
    model.layers[0].weights[0].assign(vectors)
    model.compile(loss="categorical_crossentropy", optimizer="adam", 
              metrics=['accuracy'])
    return model


def preprocessing_inputs(text,data):
    print(data.tokenizer(text[0]))
    text_normalize = [data.token(data.tokenizer(i)) for i in text]
    print(text_normalize)
    print(np.array(text_normalize))
    x= tf.keras.preprocessing.sequence.pad_sequences(text_normalize)
    print(x,x.tolist())
    return x.tolist()
if __name__ == "__main__":
    
    data = DataModel(tokenizer= Token(vocabs))
    model = create_test_model(vectors)

    inputs = np.ones(shape=(1,17))

    out = model(inputs)

    model.load_weights('./checkpoints/attention_4.h5')

    print('load_done')
    text_predict = ['giới thiệu con này đi']
    out = model.predict(preprocessing_inputs(text_predict,data))
    print(out)
    print(out.shape, data.intents[out.argmax(-1)[0]],out)

    # import tempfile
    # MODEL_DIR = './'
    # version = 1
    # export_path = os.path.join(MODEL_DIR, str(version))
    # print(export_path)
    # print('export_path = {}\n'.format(export_path))

    # tf.keras.models.save_model(
    #     model,
    #     export_path,
    #     overwrite=True,
    #     include_optimizer=True,
    #     save_format=None,
    #     signatures=None,
    #     options=None
    # )

    # print('\nSaved model:')


