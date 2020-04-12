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
	model = keras.models.load_model('./models/modelDuration.hdf5')
	# Appergiating validations, ABC and more run , pentatonic scale etc. to validate
	testList = ['sameNotes.gp5',  'AB.gp5',  'inputtedSong.gp4', 'newSong.gp5']
	# Load original durations
	with open('models/durations', 'rb') as filepath:
			originalNotes = pickle.load(filepath)
	for i in testList:
		inputNotes = readFile(i)
		x_input, y_output, durationKeys, output_size = processOriginal(originalNotes, inputNotes)
		actualList, predictionList = validate(x_input, y_output, model, durationKeys, output_size)
		display(actualList, predictionList, i)
	
# Reads validation file
def readFile(song):
	noteList = []
	myFile = './validate/'+song
	curl = guitarpro.parse(myFile)
	durationToList(noteList, curl)
	return noteList

# Processes original to get required variables for model
'''
This section retrieves help from Sigurður Skúli's guide:
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
Website was accessed between July to September 2019.

'''
def processOriginal(notes, validateNotes):
	output_size = len(set(notes))
	durationLength = 16
	durationKeys = sorted(set(item for item in notes))
	durationDictionary = dict((note, number) for number, note in enumerate(durationKeys))

	x_input = []
	y_output = []

	# Create rolling list
	for i in range(0, len(validateNotes) - durationLength, 1):
		rollingListIn = validateNotes[i:i + durationLength]
		rollingListOut = validateNotes[i + durationLength]
		x_input.append([durationDictionary[char] for char in rollingListIn])
		y_output.append(rollingListOut)

	inputLength = len(x_input)
	return x_input, y_output, durationKeys, output_size

# Predicts each duration and appends to a list
def validate(x_input, y_output, model, durationKeys, output_size):
	intDictionary = dict((number, note) for number, note in enumerate(durationKeys))
	prediction_output = []

	for i in range(len(x_input)):
		# Reshape data
		pattern = x_input[i]
		prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
		prediction_input = prediction_input / float(output_size)

		# Predict
		prediction = model.predict(prediction_input)
		index = numpy.argmax(prediction)
		result = intDictionary[index]
		# Append to new list
		prediction_output.append(result)

	return y_output, prediction_output

# Measures accuracy using varying methods
def display(actual, prediction, song):
	# From scipy documentation https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html accessed August 2019
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
	# Displays Accuracy
	print('\n~~~~~~~~\n',song,'\n~~~~~~~~~~')
	print('R Squared:', r_value**2)
	print('Slope:', slope)
	print('Standard Error:', std_err)
	avgDifference = avgDiff(actual, prediction)
	print('Average Difference:',avgDifference)
	exactFreq = exactFreq(actual, prediction)
	print('Correct Prediction Frequency:', exactFreq, '/', len(actual))
	print('Predictions: \n', prediction)
	print('\n')

	# Plots actual against prediction
	plt.plot(actual)
	plt.plot(prediction)
	plt.title('Classification Duration - ' + song)
	plt.ylabel('Duration (1/x)')
	plt.xlabel('Length of Input')
	plt.legend(['actual', 'prediction'], loc='upper left')
	plt.show()

if __name__== '__main__':
	main()
