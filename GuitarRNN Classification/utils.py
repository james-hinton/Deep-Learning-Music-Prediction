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
import keras
import guitarpro
from convertToText import *
from midiutil.MidiFile import MIDIFile

def main():
    # Load Model
    model = keras.models.load_model('./models/model.hdf5')
    # Read input file
    inputNotes = readInput()
    # Loads notes from model needed for dictionary
    with open('models/notes', 'rb') as filepath:
        originalNotes = pickle.load(filepath)

    network_input, pitchnames, n_vocab = processOriginal(originalNotes, inputNotes)
    prediction_output = predict(network_input, model, pitchnames, n_vocab)
    createSong(prediction_output)


def readInput():
    noteList = []
    # File used as input
    myFile = './input/input.gp5'
    # Parse the GP5 file
    curl = guitarpro.parse(myFile)
    # Convert the file to noteList
    convertToList(noteList, curl)
    return noteList


# Process to get network input in the same style as input for original model build
def processOriginal(notes, inputNotes):
    n_vocab = len(set(notes))
    sequence_length = 16
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []

    # Create rolling list
    for i in range(0, len(inputNotes) - sequence_length, 1):
        sequence_in = inputNotes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    numpy.set_printoptions(threshold=sys.maxsize)
    return network_input, pitchnames, n_vocab


def predict(network_input, model, pitchnames, n_vocab):
    # Create dictionary to notes
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    prediction_output = []
    # Take only last part of pattern
    pattern = network_input[-1]

    # Predicts 30 notes
    for noteIndex in range(30):
        # Shape the input
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        # Get index and then note itself
        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        # Update pattern
        pattern.append(result)
        pattern = pattern[1:len(pattern)]
    print(prediction_output)
    return prediction_output

def createSong(notes):
        # Create MIDI file with prediction
        MIDI = MIDIFile(1)
        track = 0
        time=0
        MIDI.addTrackName(track, time, "sample")
        MIDI.addTempo(track,time,120)
        channel = 0

        addNote(MIDI, track, time, channel, notes)

        with open("output.mid", 'wb') as outf:
            # Save MIDI
            MIDI.writeFile(outf)
            print('Midi output created in directory folder')

def addNote(MIDI, track, time, channel, notes):
    # Individually adds MIDI data
    for i in notes:
        time += 1
        pitch = i
        volume = 100
        duration = 1
        MIDI.addNote(track, channel, pitch, time, duration, volume)


if __name__== '__main__':
    main()
