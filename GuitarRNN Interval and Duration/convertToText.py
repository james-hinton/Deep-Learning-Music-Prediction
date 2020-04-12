import guitarpro
import math
import os
from transpose import *


# Adds data from gp file into list
def convertToList(noteList, song):
    songsNotes = []
    noteListSong = []
    # Keyword list of non-inclusive tracks
    a = ['bass', 'cal', 'drum', 'rhythm', 'piano', 'organ', 'string'
    , 'vox', 'voice', 'key', 'synth', 'sax', 'choir', 'voix',
    'orgue', 'chour', 'coros', 'violin', 'base', 'chant', 'clav', 'brass']

    # Iterate over song to get individual data
    for track in song.tracks:
        if not track.isPercussionTrack:
            trackName = track.name.lower()
            # Searches for not in keyword list
            if not any(x in trackName for x in a):
                for measure in track.measures:    
                    for voice in measure.voices:
                        for beat in voice.beats:
                            # Checks not a chord
                            if len(beat.notes) == 1:
                                for note in beat.notes:
                                    # Adds note in beat
                                    note = note.realValue
                                    note = int(note)
                                    noteListSong.append(note)
        
    noteIntervals = []
    # Work out interval between each note and add to a new list
    for i in range(len(noteListSong)):
        if i == 0:
            interval = 0
        else:
            try:
                interval = 0
                difference = noteListSong[i+1] - noteListSong[i]
                interval = interval + difference
            except:
                pass
        noteIntervals.append(interval)
    # Add to list
    noteList.append(noteIntervals)

# Finds last note used in song for use by utils
def findLastNote(song):
    songsNotes = []
    noteListSong = []
    a = ['bass', 'cal', 'drum', 'rhythm', 'piano', 'organ', 'string'
    , 'vox', 'voice', 'key', 'synth', 'sax', 'choir', 'voix',
    'orgue', 'chour', 'coros', 'violin', 'base', 'chant', 'clav', 'brass']

    for track in song.tracks:
        if not track.isPercussionTrack:
            trackName = track.name.lower()
            if not any(x in trackName for x in a):
                #print(trackName)
                for measure in track.measures:    
                    for voice in measure.voices:
                        for beat in voice.beats:
                            if len(beat.notes) == 1:
                                for note in beat.notes:
                                    note = note.realValue
                                    note = int(note)
                                    # Last note will be returned
                                    return note
                                    
        

    

# Converts from note real value to note
def valueToNote(value):
    # Modulus 12 and note is left over
    note = value % 12
    octave = (value / 12)
    octave = math.floor(octave)
    # Left over equals the note name
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

# Converts duration of each note into list
def durationToList(noteDurationList, song):
    # Keyword list of non-inclusive data
    a = ['bass', 'cal', 'drum', 'rhythm', 'piano', 'organ', 'string'
    , 'vox', 'voice', 'key', 'synth', 'sax', 'choir', 'voix',
    'orgue', 'chour', 'coros', 'violin', 'base', 'chant', 'clav', 'brass']
    # Iterate over the song to get individual data
    for track in song.tracks:
        if not track.isPercussionTrack:
            trackName = track.name.lower()
            if not any(x in trackName for x in a):
                for measure in track.measures:    
                    for voice in measure.voices:
                        for beat in voice.beats:
                            if len(beat.notes) == 1:
                                for note in beat.notes:
                                    #Add note duration
                                    noteDurationList.append(beat.duration.value)
        


