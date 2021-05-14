#!/usr/bin/env python3

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import queue
import random

#plt.close("all")

#------------------------------------------------------------------------------------------------------
#Sources for Relevant Papers And Their Authors
#All Data Cited That Was Used In This Homework

# Book: Machine Learning 
# Author: Tom Mitchell


#------------------------------------------------------------------------------------------------------

# Testing Comment for Compilation
print('Starting')

def map_converter(x):
	if x == "P":
		return 0
	if x == "E":
		return -10
	if x == "G":
		return 1000
	if x == "S":
		return -1000

MyMap = pd.read_table("Map.txt", header=None,sep='\B',engine='python') #sep=",", converters={14:income_converter,1:job_converter}, usecols=[0,1,2,12,14])
print(MyMap)
#print(MyMap.applymap(map_converter))
# Several Characters.
# S, means self.
# S, takes it's position and then updates based on choices.

def getself(mymap):
	temp = mymap.values
	temp2 = np.argwhere(temp == "S")
	return [temp2[0][1],temp2[0][0]]

def getgoal(mymap):
	temp = mymap.values
	temp2 = np.argwhere(temp == "G")
	return [temp2[0][1],temp2[0][0]]

myself = getself(MyMap)
print("\n")
print(myself)
print(getgoal(MyMap))
print("\n")

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

	return([(evaluated_left, 'Left'), (evaluated_right, 'Right'), (evaluated_up, 'Up'), (evaluated_down, 'Down')])


def val_adj(value):
	if value <= 1:
		return 1
	return value * .95

def initalizemap(mymap):
	valuemap = mymap.applymap(map_converter)
	highvaluelocation = getgoal(mymap)
	length = len(valuemap[0])
	q = queue.Queue()
	v = queue.Queue()

	#p = queue.Queue()
	q.put(highvaluelocation)
	maxvalue = 1000
	for i in range(length):
		#print(q)
		while not q.empty():
			item = q.get()
			grouping = look_around(item, valuemap)
			if grouping[0][0] == 0:
				loc = move_left(item)
				#print(loc)
				v.put(loc)
				maxvalue = val_adj(maxvalue)
				valuemap[loc[0]][loc[1]] = maxvalue
			if grouping[1][0] == 0:
				loc = move_right(item)
				#print(loc)
				v.put(loc)
				maxvalue = val_adj(maxvalue)
				valuemap[loc[0]][loc[1]] = maxvalue
			if grouping[2][0] == 0:
				loc = move_up(item)
				#print(loc)
				v.put(loc)
				maxvalue = val_adj(maxvalue)
				valuemap[loc[0]][loc[1]] = maxvalue
			if grouping[3][0] == 0:
				loc = move_down(item)
				#print(loc)
				v.put(loc)
				maxvalue = val_adj(maxvalue)
				valuemap[loc[0]][loc[1]] = maxvalue
		#q = v
		#q = queue.Queue()
		#q = copy.copy(v)
		for i in v.queue:
			#p.put(i)
			q.put(i)
	#print("Break\n")
	#for i in p.queue:
	#	print(i)
	return valuemap
#print(initalizemap(MyMap))

#print(move_left(myself))
#print(move_right(myself))
#print(move_up(myself))
#print(move_down(myself))
#print(look_around(myself,MyMap))

def basic_engine():
	InternalMap = MyMap.applymap(map_converter)
	InternalLocation = myself
	PrevousLocation = [-11,-11]
	#f = open("myfile.txt", "w")
	for x in range(50):
		paths = look_around(InternalLocation,InternalMap)
		paths.sort()
		#print(paths[-1])
		#print(InternalLocation)
		#print(PrevousLocation)
		Prevention = False
		#print(InternalMap[InternalLocation[0]][InternalLocation[1]])
		if ((InternalMap[InternalLocation[0]][InternalLocation[1]]) <= paths[-1][0]):
			#print("BADHERE\n")
			if paths[-1][1] == 'Left':
				if move_left(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_left(InternalLocation)
				else:
					Prevention = True
			if paths[-1][1] == 'Right':
				if move_right(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_right(InternalLocation)
				else:
					Prevention = True
			if paths[-1][1] == 'Up':
				if move_up(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_up(InternalLocation)
				else:
					Prevention = True
			if paths[-1][1] == 'Down':
				if move_down(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_down(InternalLocation)
				else:
					Prevention = True
		if ((InternalMap[InternalLocation[0]][InternalLocation[1]]) <= paths[-2][0]) and (Prevention == True):
			#print("HERE\n")
			if paths[-2][1] == 'Left':
				PrevousLocation = InternalLocation
				InternalLocation = move_left(InternalLocation)
			if paths[-2][1] == 'Right':
				PrevousLocation = InternalLocation
				InternalLocation = move_right(InternalLocation)
			if paths[-2][1] == 'Up':
				PrevousLocation = InternalLocation
				InternalLocation = move_up(InternalLocation)
			if paths[-2][1] == 'Down':
				PrevousLocation = InternalLocation
				InternalLocation = move_down(InternalLocation)
		elif ((InternalMap[InternalLocation[0]][InternalLocation[1]]) <= paths[-3][0]) and (Prevention == True):
			#print("HERE\n")
			if paths[-3][1] == 'Left':
				PrevousLocation = InternalLocation
				InternalLocation = move_left(InternalLocation)
			if paths[-3][1] == 'Right':
				PrevousLocation = InternalLocation
				InternalLocation = move_right(InternalLocation)
			if paths[-3][1] == 'Up':
				PrevousLocation = InternalLocation
				InternalLocation = move_up(InternalLocation)
			if paths[-3][1] == 'Down':
				PrevousLocation = InternalLocation
				InternalLocation = move_down(InternalLocation)
		elif ((InternalMap[InternalLocation[0]][InternalLocation[1]]) <= paths[-4][0]) and (Prevention == True):
			if paths[-4][1] == 'Left':
				PrevousLocation = InternalLocation
				InternalLocation = move_left(InternalLocation)
			if paths[-4][1] == 'Right':
				PrevousLocation = InternalLocation
				InternalLocation = move_right(InternalLocation)
			if paths[-4][1] == 'Up':
				PrevousLocation = InternalLocation
				InternalLocation = move_up(InternalLocation)
			if paths[-4][1] == 'Down':
				PrevousLocation = InternalLocation
				InternalLocation = move_down(InternalLocation)
		else:
			if Prevention == False and ((InternalMap[InternalLocation[0]][InternalLocation[1]]) <= paths[-1][0]) :
				if paths[-1][1] == 'Left':
					PrevousLocation = InternalLocation
					InternalLocation = move_left(InternalLocation)
				if paths[-1][1] == 'Right':
					PrevousLocation = InternalLocation
					InternalLocation = move_right(InternalLocation)
				if paths[-1][1] == 'Up':
					PrevousLocation = InternalLocation
					InternalLocation = move_up(InternalLocation)
				if paths[-1][1] == 'Down':
					PrevousLocation = InternalLocation
					InternalLocation = move_down(InternalLocation)

		Prevention = False
		#f.write('%s\n' % InternalLocation)
	#f.close()
	print(InternalLocation)

#basic_engine()
def markov_engine():
	InternalMap = initalizemap(MyMap)
	InternalLocation = myself
	PrevousLocation = [-11,-11]
	for x in range(20):
		paths = look_around(InternalLocation,InternalMap)
		paths.sort()
		#print(paths[-1][1])
		#print(InternalLocation)
		#print(PrevousLocation)
		Prevention = False
		if ((InternalMap[InternalLocation[0]][InternalLocation[1]]) <= paths[-1][0]):
			#print("BADHERE\n")
			if paths[-1][1] == 'Left':
				if move_left(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_left(InternalLocation)
				else:
					Prevention = True
			if paths[-1][1] == 'Right':
				if move_right(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_right(InternalLocation)
				else:
					Prevention = True
			if paths[-1][1] == 'Up':
				if move_up(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_up(InternalLocation)
				else:
					Prevention = True
			if paths[-1][1] == 'Down':
				if move_down(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_down(InternalLocation)
				else:
					Prevention = True
		if ((InternalMap[InternalLocation[0]][InternalLocation[1]]) <= paths[-2][0]) and (Prevention == True):
			#print("HERE\n")
			if paths[-2][1] == 'Left':
				PrevousLocation = InternalLocation
				InternalLocation = move_left(InternalLocation)
			if paths[-2][1] == 'Right':
				PrevousLocation = InternalLocation
				InternalLocation = move_right(InternalLocation)
			if paths[-2][1] == 'Up':
				PrevousLocation = InternalLocation
				InternalLocation = move_up(InternalLocation)
			if paths[-2][1] == 'Down':
				PrevousLocation = InternalLocation
				InternalLocation = move_down(InternalLocation)
		Prevention = False
	print(InternalLocation)

#markov_engine()

def temporal_calculation(mymap, current_loc, mylambda, iteration_depth, max_depth):
	# Say Max Depth = 2
	# Say iteration depth = 1
	# Say lambda = 0.5
	if iteration_depth > max_depth:
		return 0

	newlambda = (mylambda**iteration_depth)
	newq = mymap[current_loc[0]][current_loc[1]]
	iteration_depth += 1

	PrevousLocation = [-11,-11]
	paths = look_around(current_loc,mymap)
	paths.sort()
	if paths[-1][1] == 'Left':
		PrevousLocation = current_loc
		current_loc = move_left(current_loc)
	if paths[-1][1] == 'Right':
		PrevousLocation = current_loc
		current_loc = move_right(current_loc)
	if paths[-1][1] == 'Up':
		PrevousLocation = current_loc
		current_loc = move_up(current_loc)
	if paths[-1][1] == 'Down':
		PrevousLocation = current_loc
		current_loc = move_down(current_loc)

	result = temporal_calculation(mymap, current_loc, mylambda, iteration_depth, max_depth)
	if iteration_depth == 1:
		return (1 - mylambda) * ((newlambda * newq) + result)
	return (newlambda * newq) + result

def temporal_training(mymap):
	for x in range(100):
		x_cordinate = random.randrange(0,9,1)
		y_cordinate = random.randrange(0,9,1)
		InternalLocation = [x_cordinate, y_cordinate]
		mylambda = 0.5
		iteration_depth = 0
		max_depth = 2
		tempvar = mymap[InternalLocation[0]][InternalLocation[1]]
		if tempvar != 1000:
			tempvar = temporal_calculation(mymap, InternalLocation, mylambda, iteration_depth, max_depth)
		#print(tempvar)
		mymap[InternalLocation[0]][InternalLocation[1]] = tempvar

	return mymap


def randomized_training(mymap):
	for x in range(1000):
		x_cordinate = random.randrange(0,9,1)
		y_cordinate = random.randrange(0,9,1)
		InternalLocation = [x_cordinate, y_cordinate]
		PrevousLocation = [-11,-11]
		paths = look_around(InternalLocation,mymap)
		paths.sort()
		if paths[-1][1] == 'Left':
			PrevousLocation = InternalLocation
			InternalLocation = move_left(InternalLocation)
		if paths[-1][1] == 'Right':
			PrevousLocation = InternalLocation
			InternalLocation = move_right(InternalLocation)
		if paths[-1][1] == 'Up':
			PrevousLocation = InternalLocation
			InternalLocation = move_up(InternalLocation)
		if paths[-1][1] == 'Down':
			PrevousLocation = InternalLocation
			InternalLocation = move_down(InternalLocation)
		tempvar = mymap[InternalLocation[0]][InternalLocation[1]]
		tempvar += 1
		if tempvar <= -1:
			tempvar = -10
		elif tempvar == 1001:
			tempvar = 1000
		elif tempvar >= 1000:
			tempvar = 999
		mymap[InternalLocation[0]][InternalLocation[1]] = tempvar
	return mymap

def q_engine():
	InternalMap = randomized_training(initalizemap(MyMap))
	print(InternalMap)
	InternalLocation = myself
	PrevousLocation = [-11,-11]
	for x in range(20):
		paths = look_around(InternalLocation,InternalMap)
		paths.sort()
		#print(paths[-1][1])
		print(InternalLocation)
		#print(PrevousLocation)
		Prevention = False
		if ((InternalMap[InternalLocation[0]][InternalLocation[1]]) <= paths[-1][0]):
			#print("BADHERE\n")
			if paths[-1][1] == 'Left':
				if move_left(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_left(InternalLocation)
				else:
					Prevention = True
			if paths[-1][1] == 'Right':
				if move_right(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_right(InternalLocation)
				else:
					Prevention = True
			if paths[-1][1] == 'Up':
				if move_up(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_up(InternalLocation)
				else:
					Prevention = True
			if paths[-1][1] == 'Down':
				if move_down(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_down(InternalLocation)
				else:
					Prevention = True
		if ((InternalMap[InternalLocation[0]][InternalLocation[1]]) <= paths[-2][0]) and (Prevention == True):
			#print("HERE\n")
			if paths[-2][1] == 'Left':
				PrevousLocation = InternalLocation
				InternalLocation = move_left(InternalLocation)
			if paths[-2][1] == 'Right':
				PrevousLocation = InternalLocation
				InternalLocation = move_right(InternalLocation)
			if paths[-2][1] == 'Up':
				PrevousLocation = InternalLocation
				InternalLocation = move_up(InternalLocation)
			if paths[-2][1] == 'Down':
				PrevousLocation = InternalLocation
				InternalLocation = move_down(InternalLocation)
		Prevention = False
	print(InternalLocation)

#q_engine()

def temporal_engine():
	InternalMap0 = initalizemap(MyMap)
	InternalMap = temporal_training(InternalMap0)
	InternalLocation = myself
	PrevousLocation = [-11,-11]
	for x in range(20):
		paths = look_around(InternalLocation,InternalMap)
		paths.sort()
		#print(paths[-1][1])
		#print(InternalLocation)
		#print(PrevousLocation)
		Prevention = False
		if ((InternalMap[InternalLocation[0]][InternalLocation[1]]) <= paths[-1][0]):
			#print("BADHERE\n")
			if paths[-1][1] == 'Left':
				if move_left(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_left(InternalLocation)
				else:
					Prevention = True
			if paths[-1][1] == 'Right':
				if move_right(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_right(InternalLocation)
				else:
					Prevention = True
			if paths[-1][1] == 'Up':
				if move_up(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_up(InternalLocation)
				else:
					Prevention = True
			if paths[-1][1] == 'Down':
				if move_down(InternalLocation) != PrevousLocation:
					Prevention = False
					PrevousLocation = InternalLocation
					InternalLocation = move_down(InternalLocation)
				else:
					Prevention = True
		if ((InternalMap[InternalLocation[0]][InternalLocation[1]]) <= paths[-2][0]) and (Prevention == True):
			#print("HERE\n")
			if paths[-2][1] == 'Left':
				PrevousLocation = InternalLocation
				InternalLocation = move_left(InternalLocation)
			if paths[-2][1] == 'Right':
				PrevousLocation = InternalLocation
				InternalLocation = move_right(InternalLocation)
			if paths[-2][1] == 'Up':
				PrevousLocation = InternalLocation
				InternalLocation = move_up(InternalLocation)
			if paths[-2][1] == 'Down':
				PrevousLocation = InternalLocation
				InternalLocation = move_down(InternalLocation)
		Prevention = False
	print(InternalLocation)

temporal_engine()

# def main_old():
# 	gate = 0
# 	print("Loading Edges...")
# 	#myData = make_externalGraph_commonList('one-hundredth.txt')
# 	myData = make_externalGraph_commonList('newspec.txt')
# 	externalgraph = myData[0]
# 	CommonList = myData[1]
# 	while (gate != 3):
# 		print("Shakespeare Text Generation Menu")
# 		print("Option 1: Generate Seeded Text")
# 		print("Option 2: Generate From User Input")
# 		print("Type 1 or 2 To Select Option, 3 to Exit")
# 		value = input("Input Integer Now:\n")
# 		if (value == "1") or (value == "2") or (value == "3"):
# 			gate = int(value)
# 		if (gate == 1):
# 			print(iterate_text(externalgraph, CommonList, "the"))
# 		if (gate == 2):
# 			userString = input("Please Provide A User String Now:\n")#thou never see
# 			print(iterate_text_user(externalgraph, CommonList, userString))
# 		print("\n")
#main()

#Ctr+B - runs python in text editing consle.

# Standard MKDP Makrov Decision Process
# Q Learing
# Temporal Difference