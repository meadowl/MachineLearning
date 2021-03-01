#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.close("all")

#------------------------------------------------------------------------------------------------------
#Sources for Relevant Papers And Their Authors
#All Data Cited That Was Used In This Homework

#For Hungarian Chickenpox Cases Data Set
#Citation Request:

#@misc{rozemberczki2021chickenpox,
#title={Chickenpox Cases in Hungary: a Benchmark Dataset for Spatiotemporal Signal Processing with Graph Neural Networks},
#author={Benedek Rozemberczki and Paul Scherer and Oliver Kiss and Rik Sarkar and Tamas Ferenci},
#year={2021},
#eprint={2102.08100},
#archivePrefix={arXiv},
#primaryClass={cs.SI}
#}

#For Air Quality Data Set
#Citation Request:

#S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario, Sensors and Actuators B: Chemical, Volume 129, Issue 2, 22 February 2008, Pages 750-757, ISSN 0925-4005, [Web Link].
#([Web Link])

#------------------------------------------------------------------------------------------------------

print('Hello World!') 
df = pd.read_csv("AirQualityUCI_CSV.csv")
#qf = df.sort_values(by=["CO(GT)", "Time"])

qf = df#.sort_values(by=["CO(GT)", "Time"])

#for co, time in zip(qf["CO(GT)"], qf["Time"]):
#	print(co, time)

# Function p(x) has /x/ as the bound variable from user input
# h is user input for volume of hypercube h^D in D dimensions
# N-cubes on the N-data points
# x1 to xn datapoints
# lower case k is the function k(u), where u has been set to (x-xn)/h
# k(u) = if |u| <= .5 then 1 else 0

#h = .07
#user_x = 3


def gaussian_kernal(h,user_x):
	total_sum = 0
	n = 1
	for xn in zip(qf["CO(GT)"]):
		total_sum += (np.exp((np.linalg.norm(user_x - xn[0])**2)/(-2*h*h))) / ((2*np.pi*h*h)**.5)
		n += 1
	n -= 1
	return (total_sum/n)

def k2(u):
	if (u <= .3):
		return .6
	elif (u <= .6):
		return .3
	elif (u <= .8):
		return .1
	else:
		return 0

def k(u):
	if (u <= .5):
		# print("I'm 1")
		return 1
	else:
		# print("I'm 0")
		return 0

def basic_kernal(h,user_x):
	total_sum = 0
	D = 1
	n = 1
	for xn in zip(qf["CO(GT)"]):
		total_sum += (1/(h**D)) * k2((user_x - xn[0])/h)
		n += 1
	n -= 1
	return (total_sum/n)
	#estimated_value = (total_sum/n)
	#print(estimated_value)

#------------------------------------------------------------------------------------------------------
# Air Quality Data: CO(GT),PT08.S1(CO),NMHC(GT),C6H6(GT),PT08.S2(NMHC),NOx(GT),PT08.S3(NOx),NO2(GT),PT08.S4(NO2),PT08.S5(O3),T,RH,AH
#------------------------------------------------------------------------------------------------------
df = pd.read_csv("AirQualityUCI_CSV.csv")


#------------------------------------------------------------------------------------------------------
# Chickenpox Data: CO(GT),PT08.S1(CO),NMHC(GT),C6H6(GT),PT08.S2(NMHC),NOx(GT),PT08.S3(NOx),NO2(GT),PT08.S4(NO2),PT08.S5(O3),T,RH,AH
#------------------------------------------------------------------------------------------------------
df = pd.read_csv("hungary_chickenpox.csv")


#Example of Graph Formation For Data Sets
#And Calculation of Estimated Values
def set_kernal(user_x):
	return gaussian_kernal(.07, user_x)
def set_kernal2(user_x):
	return basic_kernal(.07, user_x)

vectorized_kernal = np.vectorize(set_kernal)
vectorized_kernal2 = np.vectorize(set_kernal2)

fig, ax = plt.subplots()
fig2, ax2 =plt.subplots()
# arrange is (Start, Stop, Incremental Change)
x_axis = np.arange(-202.0,15.0,0.2)
y_axis = vectorized_kernal(x_axis)
y_axis2 = vectorized_kernal2(x_axis)
ax.plot(x_axis,y_axis)
ax2.plot(x_axis,y_axis2)
ax.set(xlabel='CO(GT)', ylabel='Density', title='Simple Kernal Gaussian')
ax2.set(xlabel='CO(GT)', ylabel='Density', title='Simple Kernals Parzen Win')
fig2.savefig("test2.png")
fig.savefig("test.png")
plt.show()