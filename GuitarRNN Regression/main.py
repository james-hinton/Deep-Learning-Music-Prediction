import guitarpro
import collections
from music21 import *
from convertToText import *
from os import listdir
import math
import rnntwo
from utils import *
from midiutil.MidiFile import MIDIFile



def main(): 
    # Finds guitar pro files located in tabs
    mydir = './tabs'
    files = listdir(mydir)

    n_epochs = 60

    noteList = [] # For each note e.g. C, E ,G
    noteLengthList = [] # Length of each note e.g. 0.5 , 1.0

    # Converts each gp file into a list
    for gpfile in files:
        try:
            # If C major then skip
            curl = guitarpro.parse(mydir + '/'+ gpfile)
            songKey = curl.key.name
            if str(songKey) == "CMajor":
                os.remove(mydir + '/' +gpfile)
                print('Removed -', gpfile,'- Key not found')
            else:
                convertToList(noteList, curl)
        except:
            # Error handling
            print('FILE ERROR WITH ', gpfile)
            pass

    # Build Model
    model= rnntwo.build_model(noteList, n_epochs)

if __name__== '__main__':
    main()
