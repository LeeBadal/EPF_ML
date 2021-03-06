import pandas as pd
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.python.keras.callbacks import TensorBoard,ModelCheckpoint
## All data is normalized
#data is normalized, target class UP or DOWN depending on previsous 24h
# TO RUN TENSORBOARD: cd /afs/kth.se/home/b/a/badal/.local/lib/python3.5/site-packages/nsorboard
# then: python3 main.py --logdir=/afs/kth.se/home/b/a/badal/EPF_ML/logs

pd.set_option('display.max_columns', 30)
data = pd.read_csv("electricity-normalized.csv")
seq_len = 200
future_pred = 48
ratio_pred = "nsw"
EPOCHS = 10
BATCH_SIZE = 64
columns = ["date","period","nswprice","nswdemand","vicprice","vicdemand","class"]
NAME = "{}-SEQ-{}-EPOCHS-{}-BATCH_SIZE-{}".format(seq_len,EPOCHS,BATCH_SIZE,int(time.time()))
data = data[["date","period","nswprice","nswdemand","class"]]


def preprocess(df):

    seq_data = []
    prev_days = deque(maxlen=seq_len)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days)==seq_len:
            seq_data.append([np.array(prev_days),i[-1]])
    random.shuffle(seq_data)

    buys = []
    sells = []
    ### balancing data
    for seq,target in seq_data:
        #print(seq,target)
        if target == "DOWN":
            sells.append([seq,0])
        elif target == "UP":
            buys.append([seq,1])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys),len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    seq_data = buys+sells

    random.shuffle(seq_data)

    x = []
    y = []

    for seq,target in seq_data:
        x.append(seq)
        y.append(target)

    return np.array(x),y

## out of sample -> instead of true forwarding since we have historic data
# we use out of sample forwarding to test our model, needed for time series forecasting to not overfit


times = sorted(data.index.values)

last_5pct = times[-int(0.05*len(times))] ## all values above this number are the last 5%

validation_data = data[(data.index >= last_5pct)] ## validation is a list of the last 5% of data
main_data = data[(data.index < last_5pct)]

print(main_data.head())
print(validation_data.head())


train_x,train_y = preprocess(main_data)
validation_x,validation_y = preprocess(validation_data)

print("train data: {} validation: {}".format(len(train_x),len(validation_x)))



model = Sequential()

model.add(LSTM(128,input_shape=(train_x.shape[1:]),activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation ='relu'))
model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))


opt = tf.keras.optimizers.Adam(lr=0.00001,decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',optimizer = opt,metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"

checkpoint = ModelCheckpoint("models/{}.model".format(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max'))

history = model.fit(
train_x,train_y,
batch_size=BATCH_SIZE,
epochs=EPOCHS,
validation_data=(validation_x,validation_y),
callbacks=[tensorboard,checkpoint]
)
