import guitarpro
import collections
import os
from music21 import *
import tensorflow as tf
from convertToText import *
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


def readInput(noteList):
    # Load input file
    myFile = './input/input.gp5'
    # Parse file
    curl = guitarpro.parse(myFile)

    convertToList(noteList, curl)


def addNote(beatss, model, track, channel, time, mf, noteList, scaler):
    originalNotes = noteList
    nextNotes = []
    # Reshape data
    notes = np.array(noteList[-6:])
    notes = notes.reshape(-1,1)
    notes = scaler.transform(notes)
    # Convert to list in lists
    notes = list(notes)
    notes = [[list(note) for note in notes]]
    notes= np.array(notes)
    # Predict
    q = model.predict(notes, verbose=1)
    # Convert to integer
    nextNotes.append(scaler.inverse_transform(q))

    addedPredList = np.delete(notes[0], 0)
    addedPredList = np.append(addedPredList, q)

    def findNote(addedPredList, scaler, nextNotes, q):
        # Reshape
        addedPredList = addedPredList.reshape(-1,1)
        addedPredList = list(addedPredList)
        addedPredList = [[list(note) for note in addedPredList]]
        addedPredList = np.array(addedPredList)
        q = model.predict(addedPredList, verbose = 1)
        print('Guessed',q,'from',addedPredList)
        nextNotes.append(scaler.inverse_transform(q))
        return addedPredList, q

    # Creates song of 50 notes
    for i in range(50):
        addedPredList, q = findNote(addedPredList, scaler, nextNotes, q)
        addedPredList = np.delete(addedPredList[0],0)
        addedPredList = np.append(addedPredList, q)

    allNotes = []
    for i in originalNotes:
        allNotes.append(i)
    for i in nextNotes:
        allNotes.append(i)
    # Adds to MIDI
    for i in allNotes: 
        i = int(i)
        time += 1
        pitch = i
        volume = 100
        duration = 1
        mf.addNote(track, channel, pitch, time, duration, volume)

def createSong():
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

        # Load model and scaler from original build
        model = keras.models.load_model('./models/model.hdf5', custom_objects=dependencies)
        scalerfile = './models/scaler.sav'
        scaler = pickle.load(open(scalerfile, 'rb'))

        # Load MIDI 
        mf = MIDIFile(1)
        track = 0
        time=0
        mf.addTrackName(track, time, "sample")
        mf.addTempo(track,time,120)
        channel = 0
        beatss = 0


        noteList = []
        readInput(noteList)
        # Add predictions to MIDI
        addNote(beatss, model, track, channel,  time, mf, noteList, scaler)

        with open("output.mid", 'wb') as outf:
            # Save MIDI
            mf.writeFile(outf)
            print('Midi output created in directory folder')


def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    createSong()

if __name__== '__main__':
    
    main()
