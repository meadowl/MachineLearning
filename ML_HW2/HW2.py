#!/usr/bin/env python3

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

#plt.close("all")

#------------------------------------------------------------------------------------------------------
#Sources for Relevant Papers And Their Authors
#All Data Cited That Was Used In This Homework

#For ShakespearePlays Data Set
#Citation:

#https://www.kaggle.com/kingburrito666/shakespeare-plays/metadata
#License: Unknown
#Visibility: Public
#Dataset owner: LiamLarsen
#Last updated: 2017-04-27, Version 4

#------------------------------------------------------------------------------------------------------

# Testing Comment for Compilation
print('Starting') 

def make_externalGraph_commonList(file_name):

	ShakespeareFile = open(file_name)
	ShakespeareString = ShakespeareFile.read()
	ShakespeareFile.close()

	for char in '"-.,!:?\n':
	    ShakespeareString=ShakespeareString.replace(char,' ')
	ShakespeareString = ShakespeareString.lower()
	ShakespeareList = Counter(ShakespeareString.split())

	#ShakespeareData = pd.DataFrame.from_records(ShakespeareList, index=[0])

	#for a in ShakespeareList:
	#	print(a+" Name")

	print(len(ShakespeareList))

	#Create Matrix
	internalgraph = {}


	#ShortList = ShakespeareList.most_common(500)
	#print(ShortList)
	FullList = ShakespeareString.split()

	#for a,b in zip(ShortList, (del ShortList[0]):
	#	internalgraph[(a,b)] = 0
	for a in FullList:
		for b in FullList:
			internalgraph[(a,b)] = 0
	#Intialize every point to not Null

	#InternalList1 = FullList
	#InternalList2 = FullList
	# For some reason the above references the same list.

	InternalList1 = ShakespeareString.split()
	InternalList2 = ShakespeareString.split()
	# But this split method makes them independent.

	#print(len(InternalList1))
	#print(len(InternalList2))
	InternalList1.pop(-1) #Remove the back
	InternalList2.pop(0) #Remove the front
	#print(len(InternalList1))
	#print(len(InternalList2))

	for a,b in zip(InternalList1, InternalList2):
		internalgraph[(a,b)] += 1

	#for a,b in zip(InternalList1, InternalList2):
	#	if (internalgraph[(a,b)] > 0):
	#		print(a + " " + b)

	externalgraph = {}

	#ShortList2 = dict(ShortList)
	#CommonList = list(ShortList2.keys())
	CommonList = list(dict(ShakespeareList).keys())
	#CommonList = list(ShortList2.values())

	for a in CommonList:
		for b in CommonList:
			externalgraph[(a,b)] = [0,0]

	#Where the first is the count, the second is the initialized probability


	for a in CommonList:
		for b in CommonList:
			externalgraph[(a,b)][0] = internalgraph[(a,b)]

	#Have initiated a 500 x 500 matrix 
	#representing edges of the most common 500 words

	full_length = {}
	for a in CommonList:
		local_length = 0
		for b in CommonList:
			local_length += externalgraph[(a,b)][0]
		if (local_length == 0):
			print("Word without connections found\n")
			local_length = 1
		full_length[a] = local_length

	for a in CommonList:
		for b in CommonList:
			externalgraph[(a,b)][1] = externalgraph[(a,b)][0] / full_length[a]

	#Now ever row, has the initial chances that each word is connected to the other.
	#The probability of the edge being taken as it were.

	#for a in CommonList:
	#	for b in CommonList:
	#		if (externalgraph[(a,b)][1] > .1):
	#			print(a + " " + b)

	#print("Starting Seeded Text Generation")
	return([externalgraph,CommonList])

#for a in CommonList:
#	local_length = 0
#	for b in CommonList:
#		local_length += externalgraph[(a,b)]
#	print(local_length)

# Generate new text from the text corpus.
# Seed word chosen? "sir"

def pick_edge(commonMatrix, commonList, previousWord):
	nextWord = previousWord
	for a in commonList:
		if (commonMatrix[(previousWord, a)][1] >= commonMatrix[(previousWord, nextWord)][1]):
			nextWord = a
	return nextWord

def iterate_text(commonMatrix, commonList, seedWord):
	counter = 0
	internalString = seedWord
	internalWord = seedWord
	while (counter != 10):
		internalWord = pick_edge(commonMatrix, commonList, internalWord)
		internalString += (" " + internalWord)
		counter += 1
	return internalString

#print(iterate_text(externalgraph, CommonList, "sir"))

# Perform text prediction given a sequence of words

def rebalance_matrix_weights(commonMatrix, commonList):
	full_length = {}
	for a in commonList:
		local_length = 0
		for b in commonList:
			local_length += commonMatrix[(a,b)][0]
		if (local_length == 0):
			print("Word without connections found\n")
			local_length = 1
		full_length[a] = local_length
	for a in commonList:
		for b in commonList:
			commonMatrix[(a,b)][1] = commonMatrix[(a,b)][0] / full_length[a]
	return commonMatrix

def modify_matrix(commonMatrix, commonList, userWords):
	for char in '"-.,!:?\n':
		userWords = userWords.replace(char,' ')
	userWords = userWords.lower()
	userWordsList1 = userWords.split()
	userWordsList2 = userWords.split()
	userWordsList1.pop(-1)
	userWordsList2.pop(0)
	for a,b in zip(userWordsList1, userWordsList2):
		if a in commonList:
			if b in commonList:
				commonMatrix[(a,b)][0] += 1
	rebalancedMatrix = rebalance_matrix_weights(commonMatrix, commonList)
	return rebalancedMatrix

def iterate_text_user(commonMatrix, commonList, userWords):
	modifiedMatrix = modify_matrix(commonMatrix, commonList, userWords)
	for char in '"-.,!:?\n':
		userWords = userWords.replace(char,' ')
	userWords = userWords.lower()
	userWordsList = userWords.split()
	seedWord = userWordsList[-1]
	generatedWords = iterate_text(modifiedMatrix, commonList, seedWord)
	return(userWords + " " + generatedWords)

#print(iterate_text_user(externalgraph, CommonList, "the king died from poison"))

def main():
	gate = 0
	print("Loading Edges...")
	myData = make_externalGraph_commonList('one-hundredth.txt')
	externalgraph = myData[0]
	CommonList = myData[1]
	while (gate != 3):
		print("Shakespeare Text Generation Menu")
		print("Option 1: Generate Seeded Text")
		print("Option 2: Generate From User Input")
		print("Type 1 or 2 To Select Option, 3 to Exit")
		value = input("Input Integer Now:\n")
		gate = int(value)
		if (gate == 1):
			print(iterate_text(externalgraph, CommonList, "sir"))
		if (gate == 2):
			userString = input("Please Provide A User String Now:\n")
			print(iterate_text_user(externalgraph, CommonList, userString))
		print("\n")
main()