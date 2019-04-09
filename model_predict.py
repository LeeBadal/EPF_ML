import tensorflow as tf
from rnn_elec import preprocess
from rnn_elec import testing_data,seq_len
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
import random

def main():

    categories = ["UP", "DOWN"]


    predict_x,predict_y = preprocesstest(testing_data)
    modelpath = "models/RNN_Final-01-0.5189.model"


    model = tf.keras.models.load_model(modelpath)

    print(model.metrics_names)
    print(testing_data.head(52))
    print(testing_data.tail())


    predict_loss,predict_accuracy = model.evaluate(predict_x,predict_y,verbose=0)
    predict_single = model.predict(predict_x)

    # i= 0
    # while i < len(prediction):
    #     print(prediction[i])
    #     i +=1

    print(predict_loss,predict_accuracy)
    print(predict_single[0])
    print(predict_single[1])
    print(predict_single[2])
    print(predict_single[3])
    print(predict_single[4])
    print(predict_single[5])





def preprocesstest(df):
    seq_data = []
    prev_days = deque(maxlen=seq_len)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days)==seq_len:
            seq_data.append([np.array(prev_days),i[-1]])

    list = []
    ### balancing data
    for seq,target in seq_data:
        #print(seq,target)
        if target == "DOWN":
            list.append([seq,0])
        elif target == "UP":
            list.append([seq,1])

    x= []
    y = []
    for seq,target in list:
        x.append(seq)
        y.append(target)

    return np.array(x),np.array(y)



if __name__ == "__main__":
    main()
