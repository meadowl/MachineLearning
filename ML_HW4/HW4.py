#!/usr/bin/env python3

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

#plt.close("all")

#------------------------------------------------------------------------------------------------------
#Sources for Relevant Papers And Their Authors
#All Data Cited That Was Used In This Homework

#They were found also on Wikipedia without links to the dataset.

#[AustralianCredit](https://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval))
#[License: UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/citation_policy.html)
#Visibility: Public
#Dataset owner: (confidential)
#Last updated: N/A

#[AdultData](https://archive.ics.uci.edu/ml/datasets/adult)
#[License: UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/citation_policy.html)
#Visibility: Public
#Dataset owner: Ronny Kohavi and Barry Becker
#Last updated: 1996-05-01

#------------------------------------------------------------------------------------------------------

# Testing Comment for Compilation
print('Starting')

def map_converter(x):
	if x == "P":
		return 10
	if x == "E":
		return -10
	if x == "G":
		return 1000
	if x == "S":
		return -1000

MyMap = pd.read_table("Map.txt", header=None,sep='\B',engine='python') #sep=",", converters={14:income_converter,1:job_converter}, usecols=[0,1,2,12,14])
print(MyMap)
print(MyMap.applymap(map_converter))
# Several Characters.
# S, means self.
# S, takes it's position and then updates based on choices.

def getself(mymap):
	temp = mymap.values
	temp2 = np.argwhere(temp == "S")
	return [temp2[0][1],temp2[0][0]]

myself = getself(MyMap)
print("\n")
print(myself)

def move_left(myself):
	if (myself[0] == 0):
		return [9, myself[1]]
	temp = myself[0] - 1
	return [temp, myself[1]]

def move_right(myself):
	if (myself[0] == 9):
		return [0, myself[1]]
	temp = myself[0] + 1
	return [temp, myself[1]]

def move_up(myself):
	if (myself[1] == 0):
		return [myself[0],9]
	temp = myself[1] - 1
	return [myself[0],temp]

def move_down(myself):
	if (myself[1] == 9):
		return [myself[0],0]
	temp = myself[1] + 1
	return [myself[0],temp]

def look_around(myself,mymap):
	left_object = move_left(myself)
	right_object = move_right(myself)
	up_object = move_up(myself)
	down_object = move_down(myself)

	evaluated_left = mymap[left_object[0]][left_object[1]]
	evaluated_right = mymap[right_object[0]][right_object[1]]
	evaluated_up = mymap[up_object[0]][up_object[1]]
	evaluated_down = mymap[down_object[0]][down_object[1]]

	return([evaluated_left, evaluated_right, evaluated_up, evaluated_down])

print(move_left(myself))
print(move_right(myself))
print(move_up(myself))
print(move_down(myself))
print(look_around(myself,MyMap))

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

weights  = np.random.rand(2,3)  #[[-.3,-.1,.4],[.4,.3,-.6]]
#weights = [[.3,.3,.4],[-.5,-.9,-.4]]
inputs = [[3,3,3],[-3,-3,-3],[4,4,-4]]
#biases = [0,.5]
biases = np.random.rand(2) #[-4,-6]
expectedOutput = [1,0,1]

def my_neuron(weightList, inputList, bias, my_function):
	dotproduct = np.dot(weightList,inputList)
	fullproduct = dotproduct + bias
	return my_function(fullproduct)

def adj(x):
	return (-1 + (2*x))

def adj2(x):
	return x

def my_sigmoid(value):
	value2 = value * -1
	sigmoid = 1 / (1 + np.exp(value2))
	#return sigmoid
	if value > 0:
		return value
	return 0

#def my_sigmoid(value):
	#old = 2 * value
	#return ((2 * my_sigmoid_old(old)) - 1)
	#return np.tanh(value)

def my_derivative_sigmoid(value):
	sigmoid = my_sigmoid(value) * (1 - my_sigmoid(value))
	#return sigmoid #(1-(np.tanh(value) * np.tanh(value) ))#adj2(sigmoid)
	if value > 0:
		return 1
	return 0
#ReLu

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

#print(neuron_layer(weights,inputs,biases,sigmoid_neuron))

def simple_error(expectedValue, actualValue):
	diff = (expectedValue - actualValue) * (expectedValue - actualValue)
	return diff

def my_derivative_weight(w_value, error_value):
	return error_value

def layer1_derivative_chain(weightList, expectedValue, actualValue, Adenduminputs, maybeBias):
	dotproduct = np.dot(weightList,Adenduminputs)
	fullproduct = dotproduct + maybeBias # simple_error(expectedValue, actualValue)
	full_output = []
	sigmoid_derivative = my_derivative_sigmoid(fullproduct) #my_derivative_sigmoid(simple_error(expectedValue, actualValue))
	#for w_weight in weightList:
	for in_input in Adenduminputs:
		per_weight_derivative = sigmoid_derivative * in_input #w_weight
		full_output.append(per_weight_derivative)
	return full_output

def layer1_bias_chain(weightList, expectedValue, actualValue, Adenduminputs, maybeBias):
	dotproduct = np.dot(weightList,Adenduminputs)
	fullproduct = dotproduct + maybeBias # simple_error(expectedValue, actualValue)
	return my_derivative_sigmoid(fullproduct)

def layer1_weight_adjustment(weightList, bias, expectedValue, actualValue, Adenduminputs):
	weight_adjustments = layer1_derivative_chain(weightList, expectedValue, actualValue, Adenduminputs, bias)
	bias_adjustment = layer1_bias_chain(weightList, expectedValue, actualValue, Adenduminputs, bias)
	new_weightList = np.add(weightList, weight_adjustments) #np.add for opposite affect subtract
	new_bias = bias + bias_adjustment
	return [new_weightList, new_bias]

def training_attempt(weightList, bias, inputs, sigmoid_neuron, expectedValue):
	actualValue = singleInput_neuron_layer(weightList,inputs,bias,sigmoid_neuron)
	return layer1_weight_adjustment(weightList, bias, expectedValue, actualValue, inputs)

def multiple_training_attempts(weightLists, biases, inputLists, sigmoid_neuron, expectedValues):
	new_weightLists = []
	new_biasList = []
	actualValues = neuron_layer(weightLists,inputLists,biases,sigmoid_neuron)
	#print(actualValues)
	for weightList,bias in zip(weightLists, biases):
		for inputs, expectedValue, actualValue in zip(inputLists, expectedValues, actualValues):
			#adjustments = training_attempt(weightList, bias, inputs, sigmoid_neuron, expectedValue)
			#print(actualValue[0])
			adjustments = layer1_weight_adjustment(weightList, bias, expectedValue, actualValue[0], inputs)
			weightList = adjustments[0]
			bias = adjustments[1]
		new_biasList.append(bias)
		new_weightLists.append(weightList)
	return [new_weightLists, new_biasList]

#print(multiple_training_attempts(weights,biases, inputs, sigmoid_neuron, expectedOutput))


def bulk_training(weightLists, biases, inputLists, sigmoid_neuron, expectedValues):
	for x in range(1):
		listofnewweightsbiases = multiple_training_attempts(weightLists, biases, inputLists, sigmoid_neuron, expectedValues)
		weightLists = listofnewweightsbiases[0]
		biases = listofnewweightsbiases[1]
	return listofnewweightsbiases


def main():
	print("Starting Weights and Answer Estimation")
	print([weights, biases])
	print(neuron_layer(weights,inputs,biases,sigmoid_neuron))
	listofnewweightsbiases = bulk_training(weights,biases, inputs, sigmoid_neuron, expectedOutput)
	print("Ending Weights and Answer Estimation")
	print(listofnewweightsbiases)
	print(neuron_layer(listofnewweightsbiases[0],inputs,listofnewweightsbiases[1],sigmoid_neuron))

#main()


weightlist1 = [[.1,.1,.1],[1,1,1],[-1,-1,-1]]
biases1 = [.1,1,-1]
weightlist2 = [[.1,.1,.1],[1,1,1],[-1,-1,-1]]
biases2 = [.1,1,-1]
inputs2 = [[0,0,0],[2,2,2],[8,8,8],[10,10,10]]
outputs2 = [0,0,1,1]
weightlist3 = [[.1,.1,.1],[1,1,1],[-1,-1,-1]]
biases3 = [.1,1,-1]

testinginputs = [[0,0,0],[10,10,10]]

# def main2():
# 	print("\nStarting Weights and Answer Estimation")
# 	print([weightlist1, biases1])
# 	#print(neuron_layer(weightlist1,inputs2,biases1,sigmoid_neuron))
# 	listofnewweightsbiases = bulk_training(weightlist1,biases1, inputs2, sigmoid_neuron, outputs2)
# 	print("Ending Weights and Answer Estimation")
# 	print(listofnewweightsbiases)
# 	#print(neuron_layer(listofnewweightsbiases[0],testinginputs,listofnewweightsbiases[1],sigmoid_neuron))
# 	######################################################
# 	listofnewweightsbiases_layer1 = bulk_training(weightlist1,biases1, inputs2, sigmoid_neuron, outputs2) #We train first layer of neurons, guiding them to the correct output
# 	layer_2_inputs = neuron_layer(listofnewweightsbiases_layer1[0],inputs2,listofnewweightsbiases_layer1[1],sigmoid_neuron) #We generated the trained outputs from the first layer
# 	print(layer_2_inputs) 
# 	listofnewweightsbiases_layer2 = bulk_training(weightlist2,biases2, layer_2_inputs, sigmoid_neuron, outputs2) #We train the second layder of neurons, guiding them to correct outputs, with input from the first layers.
# 	print(neuron_layer(listofnewweightsbiases_layer2[0],layer_2_inputs,listofnewweightsbiases_layer2[1],sigmoid_neuron))
# 	#We now have re-weighted every layer, so let's calculate from the test inputs
# 	layer_1_outputs = neuron_layer(listofnewweightsbiases_layer1[0],testinginputs,listofnewweightsbiases_layer1[1],sigmoid_neuron)
# 	layer_2_outputs = neuron_layer(listofnewweightsbiases_layer2[0],layer_1_outputs,listofnewweightsbiases_layer2[1],sigmoid_neuron)
# 	print("\nMain2\n")
# 	print(layer_1_outputs)
# 	print(layer_2_outputs)

# 	#METHOD 2, CALCULATE WEIGHT2 FIRST BEFORE WEIGHT1
# 	listofnewweightsbiases_layer2_b = bulk_training(weightlist2, biases2, neuron_layer(weightlist1,inputs2,biases1,sigmoid_neuron), sigmoid_neuron, outputs2)
# 	listofnewweightsbiases_layer1_b = bulk_training(weightlist1, biases1, inputs2, sigmoid_neuron, outputs2)
# 	layer_1_outputs_b = neuron_layer(listofnewweightsbiases_layer1_b[0],testinginputs,listofnewweightsbiases_layer1_b[1],sigmoid_neuron)
# 	layer_2_outputs_b = neuron_layer(listofnewweightsbiases_layer2_b[0],layer_1_outputs_b,listofnewweightsbiases_layer2_b[1],sigmoid_neuron)
# 	print("\nMain2_Method2\n")
# 	print(layer_1_outputs_b)
# 	print(layer_2_outputs_b)
# #main2()

# #[[0.0004288400779552879, 0.00025187945036100244, 0.003143450448120872], [0.9995731936129618, 0.9993662415619804, 0.9975268101701927]]


# #Starting
# #[[0.9999689571871614, 0.9994858783552396, 0.999824211145941], [0.00012480617136898043, 0.002063990235042096, 0.0007064419683480832], [0.9999956501199976, 0.9998539523803025, 0.9999570379198915]]
# #[[0.8949324552971903, 0.8872628728978398, 0.8569860573335006], [0.6769344655059836, 0.6768937088233639, 0.6768192044658126], [0.8949548949734583, 0.8872838025937321, 0.857003787287491]]
# #Main2


# #[[6.635369485641437e-09, 1.8747912761602997e-06, 2.134217713400595e-07], [0.9999999512401009, 0.9999947488833172, 0.9999991225595163]]
# #[[0.6766466935647448, 0.6766465783572023, 0.6766465822497529], [0.894962995206227, 0.8872908256488207, 0.857009949254189]]
# #[Finished in 0.6s]


# def bulk_training_main3(weightlist1, biases1, weightlist2, biases2, weightlist3, biases3, inputs2, sigmoid_neuron, outputs2):
# 	for x in range(1):
# 		listofnewweightsbiases_layer1 = bulk_training(weightlist1, biases1, inputs2, sigmoid_neuron, outputs2)
# 		layer_2_inputs = neuron_layer(listofnewweightsbiases_layer1[0], inputs2, listofnewweightsbiases_layer1[1], sigmoid_neuron)
# 		listofnewweightsbiases_layer2 = bulk_training(weightlist2, biases2, layer_2_inputs, sigmoid_neuron, outputs2)
# 		layer_3_inputs = neuron_layer(listofnewweightsbiases_layer2[0], layer_2_inputs, listofnewweightsbiases_layer2[1], sigmoid_neuron)
# 		listofnewweightsbiases_layer3 = bulk_training(weightlist3, biases3, layer_3_inputs, sigmoid_neuron, outputs2)

# 		weightlist1 = listofnewweightsbiases_layer1[0]
# 		biases1 = listofnewweightsbiases_layer1[1]
# 		weightlist2 = listofnewweightsbiases_layer2[0]
# 		biases2 = listofnewweightsbiases_layer2[1]
# 		weightlist3 = listofnewweightsbiases_layer3[0]
# 		biases3 = listofnewweightsbiases_layer3[1]
# 	return [listofnewweightsbiases_layer1, listofnewweightsbiases_layer2, listofnewweightsbiases_layer3]


# def bulk_training_main3_reversed(weightlist1, biases1, weightlist2, biases2, weightlist3, biases3, inputs2, otherinput, sigmoid_neuron, outputs2):
# 	for x in range(1):
# 		old_inputs_layer1 = neuron_layer(weightlist1, inputs2, biases1, sigmoid_neuron)
# 		old_inputs_layer2 = neuron_layer(weightlist2, old_inputs_layer1, biases2, sigmoid_neuron)

# 		listofnewweightsbiases_layer1 = bulk_training(weightlist1, biases1, inputs2, sigmoid_neuron, old_inputs_layer1)

# 		layer_2_inputs = neuron_layer(listofnewweightsbiases_layer1[0], inputs2, listofnewweightsbiases_layer1[1], sigmoid_neuron)
# 		listofnewweightsbiases_layer2 = bulk_training(weightlist2, biases2, layer_2_inputs, sigmoid_neuron, old_inputs_layer2)

# 		layer_3_inputs = neuron_layer(listofnewweightsbiases_layer2[0], layer_2_inputs, listofnewweightsbiases_layer2[1], sigmoid_neuron)		
# 		listofnewweightsbiases_layer3 = bulk_training(weightlist3, biases3, layer_3_inputs, sigmoid_neuron, outputs2)

		
		
		

# 		weightlist1 = listofnewweightsbiases_layer1[0]
# 		biases1 = listofnewweightsbiases_layer1[1]
# 		weightlist2 = listofnewweightsbiases_layer2[0]
# 		biases2 = listofnewweightsbiases_layer2[1]
# 		weightlist3 = listofnewweightsbiases_layer3[0]
# 		biases3 = listofnewweightsbiases_layer3[1]
# 	return [listofnewweightsbiases_layer1, listofnewweightsbiases_layer2, listofnewweightsbiases_layer3]

# def main3():
# 	#listofnewweightsbiases_layer1 = bulk_training(weightlist1, biases1, inputs2, sigmoid_neuron, outputs2) #We train first layer of neurons, guiding them to the correct output
# 	#layer_2_inputs = neuron_layer(listofnewweightsbiases_layer1[0], inputs2, listofnewweightsbiases_layer1[1], sigmoid_neuron) #We generated the trained outputs from the first layer
# 	##print(layer_2_inputs) 
# 	#listofnewweightsbiases_layer2 = bulk_training(weightlist2, biases2, layer_2_inputs, sigmoid_neuron, outputs2) #We train the second layder of neurons, guiding them to correct outputs, with input from the first layers.
# 	#layer_3_inputs = neuron_layer(listofnewweightsbiases_layer2[0], layer_2_inputs, listofnewweightsbiases_layer2[1], sigmoid_neuron)
# 	##print(neuron_layer(listofnewweightsbiases_layer2[0],layer_2_inputs,listofnewweightsbiases_layer2[1],sigmoid_neuron))
# 	##We now have re-weighted every layer, so let's calculate from the test inputs
# 	#listofnewweightsbiases_layer3 = bulk_training(weightlist3, biases3, layer_3_inputs, sigmoid_neuron, outputs2)
# 	holder = bulk_training_main3(weightlist1, biases1, weightlist2, biases2, weightlist3, biases3, inputs2, sigmoid_neuron, outputs2)
# 	listofnewweightsbiases_layer1 = holder[0]
# 	listofnewweightsbiases_layer2 = holder[1]
# 	listofnewweightsbiases_layer3 = holder[2]
# 	layer_1_outputs = neuron_layer(listofnewweightsbiases_layer1[0], testinginputs,listofnewweightsbiases_layer1[1], sigmoid_neuron)
# 	layer_2_outputs = neuron_layer(listofnewweightsbiases_layer2[0], layer_1_outputs,listofnewweightsbiases_layer2[1], sigmoid_neuron)
# 	layer_3_outputs = neuron_layer(listofnewweightsbiases_layer3[0], layer_2_outputs,listofnewweightsbiases_layer3[1], sigmoid_neuron)
# 	print("\nMain3\n")
# 	print(layer_1_outputs)
# 	print(layer_2_outputs)
# 	print(layer_3_outputs)

# #main3()

# #Starting
# #Main2

# #[[0.6206406084159436, 0.6206406084159436, 0.6206406084159436], [1.0, 1.0, 0.9999440546913858]]
# #[[0.9942172289145711, 0.98355196734002, 0.6394269248280247], [0.9996620081344952, 0.9981495654010871, 0.6507087958988319]]
# #[[0.7863073352241535, 0.9967651926144144, 0.9957675985136202], [0.7880422210893022, 0.9969604726719461, 0.9960231722136363]]
# #[Finished in 0.7s]

# #Since Sigmoid, only outputs between 1 to 0.
# #When you chain them, you get only betweent those positive numbers.
# #But to get a sigmoid value between .5 and 0, you need a negative input.
# #Since you only generate positive numbers from Sigmoid, you can't ever
# # get a low sigmoid value as a result of plugging in a value from a sigmoid function.
# # Which means to chain nodes, you either have to massage the Sigmoids output to a negative and positive index
# # OR use a different method of picking out your resulting values

# #Mapping Function
# # f(x) = -1000 + 2000*x
# # f(0) = -1000
# # f(1) = 1000
# # f(.5) = 0

# def catagorizes(x):
# 	if x >= 2: # Double Average
# 		return 1
# 	return 0

# def DataProcessing(ListOfResult):
# 	average_activation = []
# 	current_sum = 0
# 	length_sum = 0
# 	for resultList in ListOfResult:
# 		current_sum = sum(resultList)
# 		length_sum = len(resultList) 
# 		average_activation.append(current_sum/length_sum)
# 		current_sum = 0
# 		length_sum = 0
# 	newlist = []
# 	#print(average_activation)
# 	for resultList,averager in zip(ListOfResult,average_activation):
# 		newlist.append(list(map((lambda x : catagorizes(x/averager)), resultList))) #In this case the theshold is 1, for a number that falls right on the average line
# 	#print(newlist)
# 	return newlist

# def DataInformation(averagedcolumns, ListOfTestValues):
# 	average_correctness = []
# 	for resultList in averagedcolumns:
# 		summer = 0
# 		for result_a, result_b in zip(resultList, ListOfTestValues):
# 			if result_a != result_b:
# 				summer += 0
# 			if result_a == result_b:
# 				summer += 1
# 		#print(summer)
# 		summer = summer / len(resultList)
# 		average_correctness.append(summer)
# 	return average_correctness


# #airQuality = pd.read_csv("australian.csv")
# #print(airQuality)

# AustralianCredit = pd.read_table("australian.dat", header=None, sep=" ", usecols=[1,2,12,13,14])
# #print(AustralianCredit)

# def income_converter(x):
# 	if  x == " <=50K":
# 		return 0
# 	return 1

# def job_converter(x):
# 	if x == " Private":
# 		return 100
# 	if x == " Federal-gov":
# 		return 200
# 	if x == " Local-gov":
# 		return 300
# 	if x == " State-gov":
# 		return 400
# 	if x == " Self-emp-not-inc":
# 		return 500
# 	return 0

# AdultIncome = pd.read_table("adult.data", header=None, sep=",", converters={14:income_converter,1:job_converter}, usecols=[0,1,2,12,14])
# #print(AdultIncome.values[1][0] + 1)

# #prepared both as column users

# # One-Hidden-Layer-Implementation
# # 16 Neurons
# # 4 random weights per Neuron

# NumberOfNodes = 2

# OneRandomWeights = np.random.rand(NumberOfNodes,4)
# OneRandomBiases = np.random.rand(NumberOfNodes)

# OneRandomWeights_b = np.random.rand(1,2)
# OneRandomBiases_b = np.random.rand(1)


# #print(OneRandomWeights)
# #print(OneRandomBiases)

# AustralianCredit_Part1 = AustralianCredit.head(300)
# AustralianCredit_Part2 = AustralianCredit.tail(300)
# #print(AustralianCredit_Part2)

# ExpectedOutputAustralian_Array = AustralianCredit_Part1.pop(14)
# #print(AustralianCredit.values)

# AustralianCreditValues = AustralianCredit_Part1.values
# ExpectedOutputAustralian = ExpectedOutputAustralian_Array.values



# ExpectedOutputAustralian_Array_Part2 = AustralianCredit_Part2.pop(14)
# TestCreditAustralian_Input = AustralianCredit_Part2.values
# TestCreditAustralian_Output = ExpectedOutputAustralian_Array_Part2.values

# def AustralianCredits():
# 	print("Starting Weights and Answer Estimation")
# 	#print([OneRandomWeights, OneRandomBiases])
# 	#print(neuron_layer(OneRandomWeights,AustralianCreditValues,OneRandomBiases,sigmoid_neuron))
# 	listofnewweightsbiases = bulk_training(OneRandomWeights,OneRandomBiases, AustralianCreditValues, sigmoid_neuron, ExpectedOutputAustralian)
# 	print("Ending Weights and Answer Estimation")
# 	#print(listofnewweightsbiases)
# 	result = neuron_layer(listofnewweightsbiases[0],TestCreditAustralian_Input,listofnewweightsbiases[1],sigmoid_neuron)
# 	columnsorted = list(zip(*result))
# 	averagedcolumns = DataProcessing(columnsorted)
# 	error_calc_values = DataInformation(averagedcolumns, TestCreditAustralian_Output)
# 	print(error_calc_values)

# 	layer_1_outputs = neuron_layer(OneRandomWeights,AustralianCreditValues,OneRandomBiases,sigmoid_neuron)
# 	listofnewweightsbiases_layer1 = bulk_training(OneRandomWeights,OneRandomBiases, AustralianCreditValues, sigmoid_neuron, layer_1_outputs)
# 	layer_2_inputs = neuron_layer(listofnewweightsbiases_layer1[0],AustralianCreditValues,listofnewweightsbiases_layer1[1],sigmoid_neuron)
# 	#print(layer_2_inputs) 
# 	listofnewweightsbiases_layer2 = bulk_training(OneRandomWeights_b,OneRandomBiases_b, layer_2_inputs, sigmoid_neuron, ExpectedOutputAustralian)
# 	layer_1_outputs = neuron_layer(listofnewweightsbiases_layer1[0],TestCreditAustralian_Input,listofnewweightsbiases_layer1[1],sigmoid_neuron)
# 	layer_2_outputs = neuron_layer(listofnewweightsbiases_layer2[0],layer_1_outputs,listofnewweightsbiases_layer2[1],sigmoid_neuron)
# 	columnsorted2 = list(zip(*layer_2_outputs))
# 	averagedcolumns2 = DataProcessing(columnsorted2)
# 	error_calc_values2 = DataInformation(averagedcolumns2, TestCreditAustralian_Output)
# 	print("\nMainPart2\n")
# 	#print(layer_1_outputs)
# 	#print(layer_2_outputs)
# 	print(error_calc_values2)


# #AustralianCredits()

# AdultIncome_Part1 = AdultIncome.tail(5000)
# AdultIncome_Part2 = AdultIncome.head(5000)
# #print(AdultIncome_Part2)


# ExpectedOutputAdultIncome_Array = AdultIncome_Part1.pop(14)
# AdultIncomeValues = AdultIncome_Part1.values
# ExpectedOutputAdultIncome = ExpectedOutputAdultIncome_Array.values

# ExpectedOutputAdultIncome_Array_Part2 = AdultIncome_Part2.pop(14)
# TestAdultIncome_Input = AdultIncome_Part2.values
# TestAdultIncome_Output = ExpectedOutputAdultIncome_Array_Part2.values

# def AdultIncomes():
# 	print("Starting Weights and Answer Estimation")
# 	#print([OneRandomWeights, OneRandomBiases])
# 	#print(neuron_layer(OneRandomWeights,AustralianCreditValues,OneRandomBiases,sigmoid_neuron))
# 	listofnewweightsbiases = bulk_training(OneRandomWeights,OneRandomBiases, AdultIncomeValues, sigmoid_neuron, ExpectedOutputAdultIncome)
# 	print("Ending Weights and Answer Estimation")
# 	#print(listofnewweightsbiases)
# 	result = neuron_layer(listofnewweightsbiases[0],TestAdultIncome_Input,listofnewweightsbiases[1],sigmoid_neuron)
# 	columnsorted = list(zip(*result))
# 	averagedcolumns = DataProcessing(columnsorted)
# 	error_calc_values = DataInformation(averagedcolumns, TestAdultIncome_Output)
# 	print(error_calc_values)

# 	layer_1_outputs = neuron_layer(OneRandomWeights,AdultIncomeValues,OneRandomBiases,sigmoid_neuron)
# 	listofnewweightsbiases_layer1 = bulk_training(OneRandomWeights,OneRandomBiases, AdultIncomeValues, sigmoid_neuron, layer_1_outputs)
# 	layer_2_inputs = neuron_layer(listofnewweightsbiases_layer1[0],AdultIncomeValues,listofnewweightsbiases_layer1[1],sigmoid_neuron)
# 	#print(layer_2_inputs) 
# 	listofnewweightsbiases_layer2 = bulk_training(OneRandomWeights_b,OneRandomBiases_b, layer_2_inputs, sigmoid_neuron, ExpectedOutputAdultIncome)
# 	layer_1_outputs = neuron_layer(listofnewweightsbiases_layer1[0],TestAdultIncome_Input,listofnewweightsbiases_layer1[1],sigmoid_neuron)
# 	layer_2_outputs = neuron_layer(listofnewweightsbiases_layer2[0],layer_1_outputs,listofnewweightsbiases_layer2[1],sigmoid_neuron)
# 	columnsorted2 = list(zip(*layer_2_outputs))
# 	averagedcolumns2 = DataProcessing(columnsorted2)
# 	error_calc_values2 = DataInformation(averagedcolumns2, TestAdultIncome_Output)
# 	print("\nMainPart2\n")
# 	#print(layer_1_outputs)
# 	#print(layer_2_outputs)
# 	print(error_calc_values2)

# #AdultIncomes()
# #main()

# #main2()

# #main3()