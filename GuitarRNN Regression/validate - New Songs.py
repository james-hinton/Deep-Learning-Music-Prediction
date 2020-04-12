import guitarpro
import collections
import os
from music21 import *
import tensorflow as tf
from convertToText import *
import scipy.stats
from os import listdir
import math
from midiutil.MidiFile import MIDIFile
from keras.models import load_model
import numpy as np
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import backend as K
import pickle
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from keras.utils import plot_model

def main():
	readFile()


def readFile():
	# Finds songs in newSongs directory
	mydir = './validate/newSongs'
	files = listdir(mydir)

	noteList = [] # For each note e.g. C, E ,G

	# Converts each gp file into a list
	for gpfile in files:
		try:
			print('Found key for ', gpfile)
			curl = guitarpro.parse(mydir + '/'+ gpfile)
			songKey = curl.key.name
			# Not C Major default files
			if str(songKey) == "CMajor":
				os.remove(mydir + '/' +gpfile)
				print('Removed -', gpfile,'- Key not found')
			else:
				# Convert to List
				convertToList(noteList, curl)
		except:
			# Error handling
			pass


	myFile = 'New Songs'
	processSong(noteList,myFile)


def processSong(noteList, myFile):
	network_input = []
	network_output = []
	sequence_length = 6
	# Create rolling list
	for i in range(0, len(noteList) - sequence_length, 1):
		# Sometimes error if never seen note before
		try:
			sequence_in = noteList[i:i + sequence_length]
			sequence_out = noteList[i + sequence_length]
			network_input.append(sequence_in)
			network_output.append(sequence_out)
		except:
			# Error handling for never before seen notes
			continue

	validate(network_input, network_output, myFile)

def validate(X, y,myFile):
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
	dependencies = {
		'r_square': r_square,
		'rmse' : rmse
	}

	# Load model and scaler
	model = keras.models.load_model('./models/model.hdf5', custom_objects=dependencies)
	scalerfile = './models/scaler.sav'
	scaler = pickle.load(open(scalerfile, 'rb'))

	prediction_output = []
	for i in range(len(X)):
		# Reshape
		npX = np.array(X[i])
		npX = npX.reshape(-1,1)
		npX = scaler.transform(npX)

		notes = list(npX)
		notes = [[list(note) for note in notes]]
		notes= np.array(notes)
		# Predict
		q = model.predict(notes)
		# Get prediction value
		q = scaler.inverse_transform(q)
		prediction_output.append(int(q))
	display(prediction_output, y,myFile)

def display(prediction, actual,myFile):
	# statistical functions
	# r-squared using scipy
	def rsquared(x, y):
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
		return slope, intercept, r_value, p_value, std_err

	def avgDiff(actual, prediction):
		diffList = []
		for i in range(len(actual)):
			diffList.append(abs(actual[i] - prediction[i]))

		avgDifference = sum(diffList) / len(diffList)
		return avgDifference

	def exactFreq(actual, prediction):
		count = 0
		for i in range(len(actual)):
			if actual[i] == prediction[i]:
				count += 1
		return count

	slope, intercept, r_value, p_value, std_err = rsquared(actual, prediction)

	# Displays results
	print('\n~~~~~~~~\n',myFile,'\n~~~~~~~~~~')
	print('R Squared:', r_value**2)
	print('Slope:', slope) 
	print('Standard Error:', std_err)
	avgDifference = avgDiff(actual, prediction)
	print('Average Difference:',avgDifference)
	exactFreq = exactFreq(actual, prediction)
	print('Correct Prediction Frequency:', exactFreq, '/', len(actual))
	print('\n')

	# plots actual results against prediction
	plt.plot(actual)
	plt.plot(prediction)
	plt.title('Regression Notes - ' +myFile)
	plt.ylabel('Note')
	plt.xlabel('Length of Input')
	plt.legend(['actual', 'prediction'], loc='upper left')
	plt.show()

	actual = []
	prediction = []


if __name__== '__main__':
	
	main()