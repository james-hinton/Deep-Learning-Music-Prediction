import mido
from mido import MidiFile, MidiTrack, Message
from keras import backend as K
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pickle
from keras.utils import plot_model

def build_model(newList, n_epochs):
	# Scales data
	scaler = MinMaxScaler(feature_range=(0,1))
	scaler.fit(np.array(newList).reshape(-1,1))
	# Shapes the notes for model
	notes = list(scaler.transform(np.array(newList).reshape(-1,1)))

	notes = [list(note) for note in notes]

	# Create input output variable
	X = []
	y = []

	sequence_length = 6

	# Generates list of lists

	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		sequence_out = notes[i + sequence_length]
		X.append(sequence_in)
		y.append(sequence_out)


	# Functions for metrics
	''' 
	These metrics have been taken from TobiGitHub's post on GitHub 
	available at: https://github.com/keras-team/keras/issues/7947
	accessed July 2019
	'''
	def r_square(y_true, y_pred):
		SS_res =  K.sum(K.square(y_true - y_pred)) 
		SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
		return ( 1 - SS_res/(SS_tot + K.epsilon()) )
	def rmse(y_true, y_pred):
	    from keras import backend
	    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

	# Build Model
	model = Sequential()
	model.add(LSTM(256, input_shape=(sequence_length, 1), return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(128, input_shape=(sequence_length, 1), return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(64, input_shape=(sequence_length, 1), return_sequences=False))
	model.add(Dropout(0.3))
	model.add(Dense(1))
	model.add(Activation('linear'))
	optimizer = Adam(lr=0.001, amsgrad=True)
	model.compile(loss='mse', optimizer=optimizer, metrics=[r_square, rmse])


	X = np.array(X)
	y = np.array(y)

	history = model.fit(X, y, 6, epochs=n_epochs, verbose=1)

	# Plot Loss
	plt.plot(history.history['loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.show()

	# plot training curve for rmse
	plt.plot(history.history['rmse'])
	plt.title('rmse')
	plt.ylabel('rmse')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.show()

	# R Squared
	plt.plot(history.history['r_square'])
	plt.title('model R^2')
	plt.ylabel('R^2')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.show()



	# Save model
	filepath = './models/model.hdf5'
	model.save(filepath)

	# Save scaler
	scalerfile = './models/scaler.sav'
	pickle.dump(scaler, open(scalerfile, 'wb'))

	return model