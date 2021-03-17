#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

plt.close("all")

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

ShakespeareFile = open('one-hundredth.txt')
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


ShortList = ShakespeareList.most_common(500)
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

for a,b in zip(InternalList1, InternalList2):
	if (internalgraph[(a,b)] > 0):
		print(a + " " + b)