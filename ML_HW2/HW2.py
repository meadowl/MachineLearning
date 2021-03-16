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

ShakespeareFile = open('alllines.txt')
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