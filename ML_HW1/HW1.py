#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.close("all")

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
		total_sum += (1/(h**D)) * k((user_x - xn[0])/h)
		n += 1
	n -= 1
	return (total_sum/n)
	#estimated_value = (total_sum/n)
	#print(estimated_value)

def set_kernal(user_x):
	return basic_kernal(.07, user_x)

vectorized_kernal = np.vectorize(set_kernal)

fig, ax = plt.subplots()
# arrange is (Start, Stop, Incremental Change)
x_axis = np.arange(-202.0,15.0,0.2)
y_axis = vectorized_kernal(x_axis)
ax.plot(x_axis,y_axis)
ax.set(xlabel='CO(GT)', ylabel='Density', title='Simple Kernal')
fig.savefig("test.png")
plt.show()