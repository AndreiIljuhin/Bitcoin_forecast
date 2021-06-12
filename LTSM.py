# most of the code is taken from https://www.youtube.com/watch?v=LI94ZkjE_w4
import pandas as pd
import datetime
from datetime import timedelta
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from typing import List
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM

def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))

# [ time, low, high, open, close, volume ],
df = pd.read_csv("bitcoinprices_db3+hours+4zapt.csv", delimiter=',', parse_dates=["time"] ,date_parser=dateparse)
df = df.iloc[::-1]
df = df[:-4]
df = df[df["time"] > df["time"].max() - timedelta(days=365 * 4)]

print(df["time"].min(), df["time"].max())

#df.plot(x="time", y="open")
#plt.show()

train_size = int(df.shape[0] * 0.9)
train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]

print(train_df.shape, val_df.shape)

print(train_df["time"].min(), train_df["time"].max()) 
print(val_df["time"].min(), val_df["time"].max())

#train_df.plot(x="time", y="low")
#val_df.plot(x="time", y="low")
#plt.show()

scaler = StandardScaler()
scaler.fit(train_df[["hour", "low", "high", "open", "close", "volume"]])

def make_dataset(
    df,
     window_size, 
     sequence_stride,
     batch_size,
     use_scaler=True,
     shuffle=True
     ):
    features = df[["hour", "low", "high", "open", "close", "volume"]].iloc[:-(window_size * sequence_stride)] #  df[["low"]]
    if use_scaler:
        features = scaler.transform(features)
    data = np.array(features, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=df[["low"]].iloc[(window_size * sequence_stride):],
      sequence_length=window_size,
      sequence_stride=sequence_stride,
      shuffle=shuffle,
      batch_size=batch_size)
    return ds

window_size = 30
sequence_stride = 1
batch_size = 50
num_epochs = 300 

train_ds = make_dataset(df=train_df, window_size=window_size, sequence_stride=sequence_stride,
                         batch_size=batch_size, use_scaler=True, shuffle=True)
val_ds  =  make_dataset(df=val_df, window_size=window_size, sequence_stride=sequence_stride, 
                      batch_size=batch_size, use_scaler=True, shuffle=True)

feature, label = next(train_ds.as_numpy_iterator())
print(feature.shape)
print(feature[0])
print(label[0])

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(5)
])

optimizer = tf.optimizers.Adam(learning_rate=0.00001)
lstm_model.compile(
    loss=tf.losses.MeanSquaredError(), # MeanSquaredError MeanAbsoluteError
    optimizer=optimizer,
    metrics=[tf.metrics.MeanAbsoluteError()]
)


#lstm_model = load_model('ltsm_300_inp_6_out_low5_window_30_batch_50_lr.00001_G5000.h5')

history = lstm_model.fit(
      train_ds, 
      epochs=num_epochs,
      validation_data=val_ds,
      verbose=2
    )

lstm_model.save('ltsm_300_inp_6_out_low_window_30_batch_50_lr.00001_G5000.h5')

f = plt.figure()
f.clear()
plt.close(f)
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.show()

lstm_model.evaluate(train_ds)
lstm_model.evaluate(val_ds)
