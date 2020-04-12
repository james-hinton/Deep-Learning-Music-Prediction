import pickle
import numpy
import sys
from os import listdir
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
	model = keras.models.load_model('./models/modelInterval.hdf5')

	# Finds all files in newSongs directory
	mydir = './validate/newSongs'
	files = listdir(mydir)

	noteList = [] # For each note e.g. C, E ,G

	# Converts each gp file into a list
	for gpfile in files:
		try:
			# Only take not C Major files
			print('Found key for ', gpfile)
			curl = guitarpro.parse(mydir + '/'+ gpfile)
			songKey = curl.key.name
			if str(songKey) == "CMajor":
				os.remove(mydir + '/' +gpfile)
				print('Removed -', gpfile,'- Key not found')
			else:
				convertToList(noteList, curl)
		except:
			print('FILE ERROR WITH ', gpfile)
			pass

	# Load original notes
	with open('models/notes', 'rb') as filepath:
			originalNotes = pickle.load(filepath)

	newNotes = []
	# Required for data shape
	for i in noteList:
		for j in i:
			newNotes.append(j)


	# for matplotlib title
	i = 'New Songs'
	inputNotes = newNotes
	x_input, y_output, intervalKeys, output_size = processOriginal(originalNotes, inputNotes)
	actualList, predictionList = validate(x_input, y_output, model, intervalKeys, output_size)
	display(actualList, predictionList, i)
	

# Processes original to get required variables for model
'''
This section retrieves help from Sigurður Skúli's guide:
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
Website was accessed between July to September 2019.

'''
def processOriginal(notes, validateNotes):
	output_size = len(set(notes))
	# Sequence length
	intervalLength = 16
	intervalKeys = sorted(set(item for item in notes))
	intervalDictionary = dict((note, number) for number, note in enumerate(intervalKeys))
	x_input = []
	y_output = []
	# Creates rolling list
	for i in range(0, len(validateNotes) - intervalLength, 1):
		rollingListIn = validateNotes[i:i + intervalLength]
		rollingListOut = validateNotes[i + intervalLength]
		try:
			x_input.append([intervalDictionary[char] for char in rollingListIn])
			y_output.append(rollingListOut)
		except:
			continue

	inputLength = len(x_input)
	return x_input, y_output, intervalKeys, output_size

# Predicts each interval and appends to a list
def validate(x_input, y_output, model, intervalKeys, output_size):
	noteDictionary = dict((number, note) for number, note in enumerate(intervalKeys))
	prediction_output = []

	for i in range(len(x_input)):
		# Takes from last rolling list
		pattern = x_input[i]
		# Prediction reshape
		prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
		prediction_input = prediction_input / float(output_size)
		# Predict
		prediction = model.predict(prediction_input)
		# Find note from Dict
		index = numpy.argmax(prediction)
		result = noteDictionary[index]
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

	# Displays accuracy
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
	plt.title('Classification Interval - ' + song)
	plt.ylabel('Note')
	plt.xlabel('Length of Input')
	plt.legend(['actual', 'prediction'], loc='upper left')
	plt.show()

if __name__== '__main__':
	main()
