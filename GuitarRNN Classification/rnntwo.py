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
from keras.layers import LSTM
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def build_model(notes, epochs):
	# Store possible output size for classifier
	output_size = len(set(notes))

	# Length of sequence for each list
	listLength = 16

	# Get note names
	noteKeys = sorted(set(item for item in notes))

	# Creates dictionary of notes to minimise overall count
	noteDictionary = dict((note, number) for number, note in enumerate(noteKeys))

	# Input and Output variable
	X = []
	y = []

	lengthNotes = len(notes)

	# Creates rolling lists with length of listLength
	# Reference: https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
	for i in range(0, lengthNotes - listLength, 1):
		rollingListIn = notes[i:i + listLength]
		X.append([noteDictionary[char] for char in rollingListIn])
		rollingListOut = notes[i + listLength]
		y.append(noteDictionary[rollingListOut])

	# Calculate for shape of data
	inputLength = len(X)

	# Reshape the list using numpy
	X = numpy.reshape(X, (inputLength, listLength, 1))

	# Stores data between 0 and 1
	X = X / float(output_size)

	# Output required to be categorical
	y = np_utils.to_categorical(y)

	# Create model
	# Reference: https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
	model = Sequential()
	model.add(LSTM(
		512,
		input_shape=(X.shape[1], X.shape[2]),
		return_sequences=True
	))
	model.add(Dropout(0.3))
	model.add(LSTM(512, return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(512))
	model.add(Dense(256))
	model.add(Dropout(0.3))
	model.add(Dense(output_size))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	# Run model
	history = model.fit(X, y, epochs=epochs, batch_size=64)

	# Save model file
	filepath = './models/model.hdf5'
	model.save(filepath)

	# Plot Accuracy
	plt.plot(history.history['acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['accuracy'], loc='upper left')
	plt.show()
	# Plot Loss
	plt.plot(history.history['loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['loss'], loc='upper left')
	plt.show()