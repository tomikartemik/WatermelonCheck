import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv1D, Activation, MaxPooling1D, Flatten, Dense, Dropout

def audio(file):
	y, sr = librosa.load(file)
	return 10
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=512, strides=3,
	padding='valid', use_bias=False, input_shape=(44100, 1), name='c1d',
	activation='relu'))
model.add(Activation('relu', input_shape=(44100, 1)))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Conv1D(32, (3)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Conv1D(64, (3)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(20)) #1))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error',
	optimizer='adam',
	metrics=['accuracy'])

