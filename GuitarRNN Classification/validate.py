import pickle
import numpy
import sys
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.models import load_model
from keras import backend as K
import keras
import scipy.stats
import guitarpro
from convertToText import *
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

def main():
	# Load model
	model = keras.models.load_model('./models/model.hdf5')
	# Appergiating validations, ABC and more run , pentatonic scale 
	testList = ['sameNotes.gp5', 'CScale.gp5', 'outOfScaleSameNotes.gp5', 'AB.gp5', 'ABC.gp5', 'ABCD.gp5', 'pentatonic.gp5', 'inputtedSong.gp4', 'newSong.gp5']

	# Load original notes
	with open('models/notes', 'rb') as filepath:
			originalNotes = pickle.load(filepath)

	# Run program for each test
	for i in testList:
		inputNotes = readFile(i)
		network_input, network_output, pitchnames, n_vocab = processOriginal(originalNotes, inputNotes)
		actualList, predictionList = validate(network_input, network_output, model, pitchnames, n_vocab)
		display(actualList, predictionList, i)
	

def readFile(song):
	# Read the file of test
	noteList = []
	myFile = './validate/'+song
	curl = guitarpro.parse(myFile)
	convertToList(noteList, curl)
	return noteList

'''
This section retrieves help from Sigurður Skúli's guide:
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
Website was accessed between July to September 2019.

'''
def processOriginal(notes, validateNotes):
	# Process original to model
	n_vocab = len(set(notes))
	sequence_length = 16
	pitchnames = sorted(set(item for item in notes))
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

	network_input = []
	network_output = []

	for i in range(0, len(validateNotes) - sequence_length, 1):
		sequence_in = validateNotes[i:i + sequence_length]
		sequence_out = validateNotes[i + sequence_length]
		network_input.append([note_to_int[char] for char in sequence_in])
		network_output.append(sequence_out)

	n_patterns = len(network_input)
	numpy.set_printoptions(threshold=sys.maxsize)
	return network_input, network_output, pitchnames, n_vocab

def validate(network_input, network_output, model, pitchnames, n_vocab):
	# Create note dictionary
	int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
	prediction_output = []

	for i in range(len(network_input)):
		# Reshape data
		pattern = network_input[i]
		prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
		prediction_input = prediction_input / float(n_vocab)
		# Get result from dictionary
		prediction = model.predict(prediction_input)
		index = numpy.argmax(prediction)
		result = int_to_note[index]
		prediction_output.append(result)
	

	return network_output, prediction_output

def display(actual, prediction, song):
	# Load statistical functions
	# Calculates the r-squared value
	def rsquared(x, y):
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
		return slope, intercept, r_value, p_value, std_err

	# Calculates the average difference
	def avgDiff(actual, prediction):
		diffList = []
		for i in range(len(actual)):
			diffList.append(abs(actual[i] - prediction[i]))

		avgDifference = sum(diffList) / len(diffList)
		return avgDifference

	# Times there was an exact prediction
	def exactFreq(actual, prediction):
		count = 0
		for i in range(len(actual)):
			if actual[i] == prediction[i]:
				count += 1
		return count

	slope, intercept, r_value, p_value, std_err = rsquared(actual, prediction)

	# Displays data
	print('\n~~~~~~~~\n',song,'\n~~~~~~~~~~')
	print('R Squared:', r_value**2)
	print('Slope:', slope) 
	print('Standard Error:', std_err)
	avgDifference = avgDiff(actual, prediction)
	print('Average Difference:',avgDifference)
	exactFreq = exactFreq(actual, prediction)
	print('Correct Prediction Frequency:', exactFreq, '/', len(actual))
	print('\n')

	# Plots actual against prediction
	plt.plot(actual)
	plt.plot(prediction)
	plt.title('Classification Notes - ' + song)
	plt.ylabel('Note')
	plt.xlabel('Length of Input')
	plt.legend(['actual', 'prediction'], loc='upper left')
	plt.show()

# Run program
if __name__== '__main__':
	main()
