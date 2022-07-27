import numpy as np
from test import window_dataset, seq2seq_window_dataset,model_forecast
import tensorflow as tf
import pandas as pd
import io
keras = tf.keras
from numpy import genfromtxt
my_data = genfromtxt(r'C:\Users\MAAZ PATEL\PycharmProjects\untitled2\Brent Spot Price.csv', delimiter=',')
series = my_data

split_time = 300
time = np.arange(4*99)
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)
window_size = 30
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
train_set = window_dataset(x_train, window_size, batch_size=128)
valid_set = window_dataset(x_valid, window_size, batch_size=128)

model = keras.models.Sequential([
  keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
  keras.layers.SimpleRNN(100, return_sequences=True),
  keras.layers.SimpleRNN(100),
  keras.layers.Dense(1),
  keras.layers.Lambda(lambda x: x * 200.0)
])
optimizer = keras.optimizers.SGD(lr=1.5e-6, momentum=0.9)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
early_stopping = keras.callbacks.EarlyStopping(patience=50)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    "my_checkpoint", save_best_only=True)
model.fit(train_set, epochs=100,
          validation_data=valid_set,
          callbacks=[early_stopping, model_checkpoint])
model.save(r'C:\Users\MAAZ PATEL\PycharmProjects\untitled2')
#model = keras.models.load_model("my_checkpoint")
rnn_forecast = model_forecast(
    model,
    series[split_time - window_size:-1],
    window_size)[:, 0]
m = tf.keras.metrics.MeanAbsoluteError()
m.update_state(x_valid, rnn_forecast)
print(m.result().numpy())
