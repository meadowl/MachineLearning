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

# Testing Comment for Compilation
print('Starting') 

# Function p(x) has /x/ as the bound variable from user input
# h is user input for volume of hypercube h^D in D dimensions
# N-cubes on the N-data points
# x1 to xn datapoints
# lower case k is the function k(u), where u has been set to (x-xn)/h
# k(u) = if |u| <= .5 then 1 else 0

#h = .07
#user_x = 3

# Gaussian method subsituted in for k(u)
# generates the gaussian method specifically for kernal density mixture modeling

def generic_gaussian_kernal(h,user_x,data_set,data_label):
	total_sum = 0
	n = 1
	for xn in zip(data_set[data_label]):
		total_sum += (np.exp((np.linalg.norm(user_x - xn[0])**2)/(-2*h*h))) / ((2*np.pi*h*h)**.5)
		n += 1
	n -= 1
	return (total_sum/n)

# Defined in such a way that for a k_function that takes the difference from the user's x_value against a list's x_vaulues
# That it will process as long as the k_function is of a basic version, such as k2 down further below
def generic_basic_kernal(h,user_x,k_funct,data_set,data_label):
	total_sum = 0
	D = 1
	n = 1
	for xn in zip(data_set[data_label]):
		total_sum += (1/(h**D)) * k_funct((user_x - xn[0])/h)
		n += 1
	n -= 1
	return (total_sum/n)

# Two basic kernal functions for weighting
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
#------------------------------------------------------------------------------------------------------

#Outdated Imported Learning Attempts
#Old Non-Generic Kernal Methods for Reference

#df = pd.read_csv("AirQualityUCI_CSV.csv")
#qf = df.sort_values(by=["CO(GT)", "Time"])

#qf = df#.sort_values(by=["CO(GT)", "Time"])

#for co, time in zip(qf["CO(GT)"], qf["Time"]):
#	print(co, time)

#def gaussian_kernal(h,user_x):
#	total_sum = 0
#	n = 1
#	for xn in zip(qf["CO(GT)"]):
#		total_sum += (np.exp((np.linalg.norm(user_x - xn[0])**2)/(-2*h*h))) / ((2*np.pi*h*h)**.5)
#		n += 1
#	n -= 1
#	return (total_sum/n)

# def basic_kernal(h,user_x):
#	total_sum = 0
#	D = 1
#	n = 1
#	for xn in zip(qf["CO(GT)"]):
#		total_sum += (1/(h**D)) * k2((user_x - xn[0])/h)
#		n += 1
#	n -= 1
#	return (total_sum/n)
	#estimated_value = (total_sum/n)
	#print(estimated_value)

#------------------------------------------------------------------------------------------------------
# Air Quality Data: CO(GT),PT08.S1(CO),NMHC(GT),C6H6(GT),PT08.S2(NMHC),NOx(GT),PT08.S3(NOx),NO2(GT),PT08.S4(NO2),PT08.S5(O3),T,RH,AH
#------------------------------------------------------------------------------------------------------
airQuality = pd.read_csv("AirQualityUCI_CSV.csv")

def cogt(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"CO(GT)")
def pt08s1co(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"PT08.S1(CO)")
def nmhcgt(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"NMHC(GT)")
def c6h6gt(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"C6H6(GT)")
def pt08s2nmhc(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"PT08.S2(NMHC)")
def noxgt(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"NOx(GT)")
def pt08s3nox(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"PT08.S3(NOx)")
def no2gt(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"NO2(GT)")
def pt08s4no2(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"PT08.S4(NO2)")
def pt08s503(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"PT08.S5(O3)")
def t(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"T")
def rh(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"RH")
def ah(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"AH")

#Test
print(pt08s2nmhc(3))

#------------------------------------------------------------------------------------------------------
# Chickenpox Data: BUDAPEST,BARANYA,BACS,BEKES,BORSOD,CSONGRAD,FEJER,GYOR,HAJDU,HEVES,JASZ,KOMAROM,NOGRAD,PEST,SOMOGY,SZABOLCS,TOLNA,VAS,VESZPREM,ZALA
#------------------------------------------------------------------------------------------------------
chickenPox = pd.read_csv("hungary_chickenpox.csv")

def budapest(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"BUDAPEST")
def baranya(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"BARANYA")
def bacs(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"BACS")
def bekes(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"BEKES")
def borsod(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"BORSOD")
def csongrad(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"CSONGRAD")
def fejer(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"FEJER")
def gyor(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"GYOR")
def hajdu(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"HAJDU")
def heves(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"HEVES")
def jasz(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"JASZ")
def komarom(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"KOMAROM")
def nograd(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"NOGRAD")
def pest(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"PEST")
def somogy(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"SOMOGY")
def szabolcs(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"SZABOLCS")
def tolna(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"TOLNA")
def vas(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"VAS")
def veszprem(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"VESZPREM")
def zala(user_x):
	return generic_gaussian_kernal(.07, user_x,chickenPox,"ZALA")

#Test
print(str(nograd(12)))

#------------------------------------------------------------------------------------------------------
#Example of Graph Formation For Data Sets
#And Calculation of Estimated Values
def set_kernal(user_x):
	return generic_gaussian_kernal(.07, user_x,airQuality,"CO(GT)")
def set_kernal2(user_x):
	return generic_basic_kernal(.07, user_x,k2,airQuality,"CO(GT)")

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