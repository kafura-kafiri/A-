from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)

number_of_types = 10
seq = 1
output = 1

model = Sequential()
model.add(LSTM(100, input_shape=(seq, output * 2 * seq + number_of_types)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
