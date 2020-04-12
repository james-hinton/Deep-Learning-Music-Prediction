import guitarpro
import math
import os
from transpose import *


# Adds data from gp file into list
def convertToList(noteList, song):
    # Detect Song Key
    songKey = song.key.name
    songsNotes = []
    # Detect semitone change required
    semitone = transposeBy(songKey)
    print(songKey, ' to be transposed to CMajor by', semitone, 'semitones')
    # Keyword list of non-tracks
    a = ['bass', 'cal', 'drum', 'rhythm', 'piano', 'organ', 'string'
    , 'vox', 'voice', 'key', 'synth', 'sax', 'choir', 'voix',
    'orgue', 'chour', 'coros', 'violin', 'base', 'chant', 'clav']

    # Iterate to find individual data
    for track in song.tracks:
        # Makes sure not a drum track
        if not track.isPercussionTrack:
            trackName = track.name.lower()
            # Searches keyword list
            if not any(x in trackName for x in a):
                for measure in track.measures:    
                    for voice in measure.voices:
                        for beat in voice.beats:
                            # Checks to make sure it is not a chord
                            if len(beat.notes) == 1:
                                for note in beat.notes:
                                    # Process the note in accordance to the semitone split
                                    convertednote = process(track, measure, voice, beat, note, semitone)
                                    # Add to list
                                    songsNotes.append(convertednote)
                                    noteList.append(convertednote)
        
    songsString = []                    
    for i in songsNotes:
        songsString.append(valueToNote(i))



# Converts from note real value to 
def valueToNote(value):
    # Modulus 12 to get remainder
    note = value % 12
    # Find octave
    octave = (value / 12)
    octave = math.floor(octave)
    # Remainder dictates the note
    if note == 0:
        return 'C' + str(octave)
    elif note == 1:
        return 'C#'  + str(octave)
    elif note == 2:
        return 'D'  + str(octave)
    elif note == 3:
        return 'D#'+ str(octave)
    elif note == 4:
        return 'E'+ str(octave)
    elif note == 5:
        return 'F'+ str(octave)
    elif note == 6:
        return 'F#'+ str(octave)
    elif note == 7:
        return 'G'+ str(octave)
    elif note == 8:
        return 'G#'+ str(octave)
    elif note == 9:
        return 'A'+ str(octave)
    elif note == 10:
        return 'A#'+ str(octave)
    elif note == 11:
        return 'B'+ str(octave)

