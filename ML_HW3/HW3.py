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

	print("Unique Words: " + str(len(ShakespeareList)))

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
	while (counter != 20):
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

def sub_iterate_text(commonMatrix, commonList, seedWord):
	counter = 0
	internalString = ""
	internalWord = seedWord
	while (counter != 20):
		internalWord = pick_edge(commonMatrix, commonList, internalWord)
		internalString += (" " + internalWord)
		counter += 1
	return internalString

def iterate_text_user(commonMatrix, commonList, userWords):
	modifiedMatrix = modify_matrix(commonMatrix, commonList, userWords)
	for char in '"-.,!:?\n':
		userWords = userWords.replace(char,' ')
	userWords = userWords.lower()
	userWordsList = userWords.split()
	seedWord = userWordsList[-1]
	generatedWords = sub_iterate_text(modifiedMatrix, commonList, seedWord)
	return(userWords + generatedWords)

#print(iterate_text_user(externalgraph, CommonList, "the king died from poison"))

def main_old():
	gate = 0
	print("Loading Edges...")
	#myData = make_externalGraph_commonList('one-hundredth.txt')
	myData = make_externalGraph_commonList('newspec.txt')
	externalgraph = myData[0]
	CommonList = myData[1]
	while (gate != 3):
		print("Shakespeare Text Generation Menu")
		print("Option 1: Generate Seeded Text")
		print("Option 2: Generate From User Input")
		print("Type 1 or 2 To Select Option, 3 to Exit")
		value = input("Input Integer Now:\n")
		if (value == "1") or (value == "2") or (value == "3"):
			gate = int(value)
		if (gate == 1):
			print(iterate_text(externalgraph, CommonList, "the"))
		if (gate == 2):
			userString = input("Please Provide A User String Now:\n")#thou never see
			print(iterate_text_user(externalgraph, CommonList, userString))
		print("\n")
#main()

#Ctr+B - runs python in text editing consle.

# List of weights [.3,.3,.4]
# List of inputs [1,2,3]

weights  = [[.3,.1,.9],[-.3,-.5,-.9]]
#weights = [[.3,.3,.4],[-.5,-.9,-.4]]
inputs = [[1,1,1],[0,0,0]]
#biases = [0,.5]
biases = [0,1]
expectedOutput = [1,0]

def my_neuron(weightList, inputList, bias, my_function):
	dotproduct = np.dot(weightList,inputList)
	fullproduct = dotproduct + bias
	return my_function(fullproduct)

def my_sigmoid(value):
	value2 = value * -1
	sigmoid = 1 / (1 + np.exp(value2))
	return sigmoid

def sigmoid_neuron(weightList, inputList, bias):
	return my_neuron(weightList, inputList, bias, my_sigmoid)

def singleInput_neuron_layer(EveryNeurons_weightList, inputList, EveryNeurons_bias, Individual_neuron):
	full_output = []
	for weightList,bias in zip(EveryNeurons_weightList, EveryNeurons_bias):
		single_output = Individual_neuron(weightList, inputList, bias)
		full_output.append(single_output)
	return full_output

def neuron_layer(EveryNeurons_weightList, Every_inputList, EveryNeurons_bias, Individual_neuron):
	full_output = []
	for inputList in Every_inputList:
		single_output = singleInput_neuron_layer(EveryNeurons_weightList, inputList, EveryNeurons_bias, Individual_neuron)
		full_output.append(single_output)
	return full_output

print(neuron_layer(weights,inputs,biases,sigmoid_neuron))

def simple_error(expectedValue, actualValue):
	diff = (expectedValue - actualValue)
	return diff


def my_derivative_sigmoid(value):
	return my_sigmoid(value) * (1 - my_sigmoid(value))

def my_derivative_weight(w_value, error_value):
	return error_value

def layer1_derivative_chain(weightList, expectedValue, actualValue):
	#dotproduct = np.dot(weightList,inputList)
	#fullproduct = dotproduct + bias
	full_output = []
	sigmoid_derivative = my_derivative_sigmoid(simple_error(expectedValue, actualValue))
	for w_weight in weightList:
		per_weight_derivative = sigmoid_derivative * w_weight
		full_output.append(.1 * per_weight_derivative)
	return full_output

def layer1_bias_chain(expectedValue, actualValue):
	return .1 * my_derivative_sigmoid(simple_error(expectedValue, actualValue))

def layer1_weight_adjustment(weightList, bias, expectedValue, actualValue):
	weight_adjustments = layer1_derivative_chain(weightList, expectedValue, actualValue)
	bias_adjustment = layer1_bias_chain(expectedValue, actualValue)
	new_weightList = np.add(weightList, weight_adjustments) #np.add for opposite affect
	new_bias = bias + bias_adjustment
	return [new_weightList, new_bias]

def training_attempt(weightList, bias, inputs, sigmoid_neuron, expectedValue):
	actualValue = singleInput_neuron_layer(weightList,inputs,bias,sigmoid_neuron)
	return layer1_weight_adjustment(weightList, bias, expectedValue, actualValue)

def multiple_training_attempts(weightLists, biases, inputLists, sigmoid_neuron, expectedValues):
	new_weightLists = []
	new_biasList = []
	actualValues = neuron_layer(weightLists,inputLists,biases,sigmoid_neuron)
	#print(actualValues)
	for weightList,bias in zip(weightLists, biases):
		for inputs, expectedValue, actualValue in zip(inputLists, expectedValues, actualValues):
			#adjustments = training_attempt(weightList, bias, inputs, sigmoid_neuron, expectedValue)
			#print(actualValue[0])
			adjustments = layer1_weight_adjustment(weightList, bias, expectedValue, actualValue[0])
			weightList = adjustments[0]
			bias = adjustments[1]
		new_biasList.append(bias)
		new_weightLists.append(weightList)
	return [new_weightLists, new_biasList]

print(multiple_training_attempts(weights,biases, inputs, sigmoid_neuron, expectedOutput))


def bulk_training(weightLists, biases, inputLists, sigmoid_neuron, expectedValues):
	for x in range(100):
		listofnewweightsbiases = multiple_training_attempts(weightLists, biases, inputLists, sigmoid_neuron, expectedValues)
		weightLists = listofnewweightsbiases[0]
		biases = listofnewweightsbiases[1]
	return listofnewweightsbiases


def main():
	listofnewweightsbiases = bulk_training(weights,biases, inputs, sigmoid_neuron, expectedOutput)
	print(listofnewweightsbiases)
	print(neuron_layer(listofnewweightsbiases[0],inputs,listofnewweightsbiases[1],sigmoid_neuron))

main()
