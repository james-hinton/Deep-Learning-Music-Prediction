'''
The idea of preprocessing and dictionary conversion in rolling lists retrieves help from Sigurður Skúli's guide:
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
Website was accessed between July to September 2019.

'''

import glob
import pickle
import numpy
import sys
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import os

def train_network(intervals, epochs):
	# Store possible output size for classifier
	output_size = len(set(intervals))

	# Length of sequence for each list
	intervalLength = 16

	# Get interval names
	intervalKeys = sorted(set(item for item in intervals))

	# create a dictionary of intervals to minimise overall count	
	intervalDictionary = dict((note, number) for number, note in enumerate(intervalKeys))

	# Input and Output variable
	X = []
	y = []

	# Creates rolling lists with length of interval length
	# Reference: https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
	for i in range(0, len(intervals) - intervalLength, 1):
		rollingListIn = intervals[i:i + intervalLength]
		rollingListOut = intervals[i + intervalLength]
		X.append([intervalDictionary[char] for char in rollingListIn])
		y.append(intervalDictionary[rollingListOut])

	# Calculate for shape of data
	inputLength = len(X)

	# Reshape list
	X = numpy.reshape(X, (inputLength, intervalLength, 1))
	
	# Stores data between 0 and 1
	X = X / float(output_size)

	# Output required to be categorical
	y = np_utils.to_categorical(y)


	# Create model
	model = Sequential()
	model.add(LSTM(
		256,
		input_shape=(X.shape[1], X.shape[2]),
		return_sequences=True
	))
	model.add(Dropout(0.3))
	model.add(LSTM(128, return_sequences=False))
	model.add(Dropout(0.3))
	model.add(Dense(output_size))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	# Run model
	history = model.fit(X, y, epochs=epochs, batch_size=64)

	# Save model file
	filepath = './models/modelInterval.hdf5'
	model.save(filepath)

	# Plot Accuracy
	plt.plot(history.history['acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.show()
	# Plot Loss
	plt.plot(history.history['loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.show()