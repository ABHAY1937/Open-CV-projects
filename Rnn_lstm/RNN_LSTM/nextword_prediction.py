import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle #for reading the binary lines 

with open('/home/abhay/Documents/DL Projects/RNN_LSTM/tokenizer.pickle','rb') as handle:
    mytokenizer = pickle.load(handle)

model = tf.keras.models.load_model('/home/abhay/Documents/DL Projects/RNN_LSTM/word_generation_model.h5')

input_text = 'india'
predict_next_words = 10

for i in range(predict_next_words):
    token_list = mytokenizer.texts_to_sequences([input_text])[0] #tokenize
    token_list = pad_sequences([token_list],maxlen = model.input_shape[1],padding = 'pre') #padding
    predicted = np.argmax(model.predict(token_list),axis = -1) #max_value find out== np.argmax , axis =-1 taking last Access
    #number to word ====>
    output_word = mytokenizer.index_word[predicted[0]]
    input_text += " " + output_word
print(input_text)
