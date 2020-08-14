import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot 

sents=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

voc_size = 10000;

onehot_repr = [one_hot(words,voc_size) for words in sent]

print(onehot_repr)

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

sent_len = 8
embedded_docs = pad_sequences(onehot_repr,padding = 'pre' , maxlen = sent_len)

print(embedded_docs)


#model----

model = Sequential()
model.add(Embedding(voc_size,10,input_length = sent_len))
model.compile('adam','mse')

model.summary()

print(model.predict(embedded_docs))