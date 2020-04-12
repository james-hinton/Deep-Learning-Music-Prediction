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
from os import listdir

def main():
	# Load model
	model = keras.models.load_model('./models/modelDuration.hdf5')


	mydir = './validate/newSongs'
	files = listdir(mydir)

	noteList = [] # For each note e.g. C, E ,G

	# Converts each gp file into a list
	for gpfile in files:
		try:
			print('Found key for ', gpfile)
			curl = guitarpro.parse(mydir + '/'+ gpfile)
			songKey = curl.key.name
			if str(songKey) == "CMajor":
				os.remove(mydir + '/' +gpfile)
				print('Removed -', gpfile,'- Key not found')
			else:
				durationToList(noteList, curl)
		except:
			print('FILE ERROR WITH ', gpfile)
			pass

	# Load original durations
	with open('models/durations', 'rb') as filepath:
			originalNotes = pickle.load(filepath)

	i = 'New Notes'
	inputNotes = noteList
	x_input, y_output, durationKeys, output_size = processOriginal(originalNotes, inputNotes)
	actualList, predictionList = validate(x_input, y_output, model, durationKeys, output_size)
	display(actualList, predictionList, i)
	

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
	# Creates rolling list with input and output
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
		# Predict data
		prediction = model.predict(prediction_input)
		index = numpy.argmax(prediction)
		result = intDictionary[index]
		# Predictions to new list
		prediction_output.append(result)

	return y_output, prediction_output

# Measures accuracy using varying methods
def display(actual, prediction, song):
	# From scipy documentation https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html accessed August 2019
	def rsquared(x, y):
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
		return slope, intercept, r_value, p_value, std_err
	# Calculate average distance
	def avgDiff(actual, prediction):
		diffList = []
		for i in range(len(actual)):
			diffList.append(abs(actual[i] - prediction[i]))

		avgDifference = sum(diffList) / len(diffList)
		return avgDifference
	# Calculate how many times it was exact
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

	# Plot actual against prediction
	plt.plot(actual)
	plt.plot(prediction)
	plt.title('Classification Duration - ' + song)
	plt.ylabel('Duration (1/x)')
	plt.xlabel('Length of Input')
	plt.legend(['actual', 'prediction'], loc='upper left')
	plt.show()

if __name__== '__main__':
	main()
