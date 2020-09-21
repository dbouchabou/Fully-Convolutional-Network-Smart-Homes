#!/usr/bin/env python3


import os

import argparse

import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_raw_dataset(input_file):

	df = pd.read_csv(input_file,sep="\t",header=None,names=["date","time","sensor","value","activity","log"])

	return df

def clean_and_prepare(df):

	#rempli les valeurs NaN de la colonne log avec la valeur précédente
	df.log = df.log.fillna(method='ffill')

	#rempli les valeur NaN de la colonne activity avec lavaleur de la colonne log
	df['activity'] = df['activity'].fillna(df['log'])

	df['activity'] = df['activity'].replace("end", "Other")

	df['activity'] = df['activity'].fillna("Other")

	df['activity'] = df['activity'].replace("begin", None)

	return df

def save_activity_dict(df, input_file):

	if "milan" in input_file:
		filename = "milan_activity_list.pickle"

	if "aruba" in input_file:
		filename = "aruba_activity_list.pickle"

	activities = df.activity.unique()
	activities.sort()

	dictActivities = {}
	for i, activity in enumerate(activities):
		dictActivities[activity] = i

	pickle_out = open(filename,"wb")
	pickle.dump(dictActivities, pickle_out)
	pickle_out.close()


def generate_sentence(df2):
    
    sentence = ""
    
    val = "" 

    #extract sensors list
    sensors = df2.sensor.values

    #extract values
    values = df2.value.values

    #iterate on sensors list
    for i in range(len(sensors)):

        val = values[i]

        if i == len(sensors) - 1:
            sentence += "{}{}".format(sensors[i],val)
        else:
            sentence += "{}{} ".format(sensors[i],val)

    return sentence


def segment_activities(df):

	activitiesSeq = []

	ponentialIndex = df.activity.ne(df.activity.shift())

	ii = np.where(ponentialIndex == True)[0]

	for i,end in enumerate(ii):
	    if i > 0 :
	        
	        dftmp = df[ii[i-1]:end]

	        activitiesSeq.append(dftmp)
          

	return activitiesSeq


def slidingWindow(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
    
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    
    numOfChunks = int(((len(sequence)-winSize)/step)+1)
    
    # Do the work
    if winSize > len(sequence):
        yield sequence[0:len(sequence)]
    else:
        for i in range(0,numOfChunks*step,step):
            yield sequence[i:i+winSize]

def sequencesToSentences(activitySequences):
	sentences = []
	label_sentences = []

	for i in range(len(activitySequences)):

		sentence = generate_sentence(activitySequences[i])

		sentences.append(sentence)
		label_sentences.append(activitySequences[i].activity.values[0])

	return sentences, label_sentences

if __name__ == '__main__':

	p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
	p.add_argument('--i', dest='input', action='store', default='', help='input', required = True)
	p.add_argument('--o', dest='output', action='store', default='', help='output', required = True)
	p.add_argument('--w', dest='winSize', action='store', default='', help='windows size', required = True)

	args = p.parse_args()

	input_file = str(args.input)
	output_file = str(args.output)
	winSize = int(args.winSize)


	print("STEP 1: Load dataset")
	df = load_raw_dataset(input_file)

	print("STEP 2: prepare dataset")
	df = clean_and_prepare(df)


	save_activity_dict(df,input_file)

	## Segment dataset in sequence of activity ##
	print("STEP 3: segment dataset in sequence of activity")
	activitySequences = segment_activities(df)

	## Transform sequences of activity in sentences ##
	print("STEP 4: transform sequences of activity in sentences")
	sentences, label_sentences = sequencesToSentences(activitySequences)

	## Indexization ##
	print("STEP 5: sentences indexization")
	tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
	tokenizer.fit_on_texts(sentences)

	word_index = tokenizer.word_index

	indexed_sentences = tokenizer.texts_to_sequences(sentences)

	## Split in sliding windows ##
	print("STEP 6: split indexed sentences in sliding windows")
	X_windowed = []
	Y_windowed = []

	step = 1

	for i,s in enumerate(indexed_sentences):
		chunks = slidingWindow(s,winSize,step)
		for chunk in chunks:
			X_windowed.append(chunk)
			Y_windowed.append(label_sentences[i])


	## Pad windows ##
	print("STEP 7: pad sliding windows")
	padded_windows = pad_sequences(X_windowed)

	Y_windowed = np.array(Y_windowed)

	## Save files ##
	print("STEP 8: save sliding windows and labels")
	np.save("{}_{}_padded_x.npy".format(output_file,winSize), padded_windows)
	np.save("{}_{}_padded_y.npy".format(output_file,winSize), Y_windowed)