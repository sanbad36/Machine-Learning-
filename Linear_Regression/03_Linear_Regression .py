#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Sanket Badjate...
from math import sqrt
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt



class Fromscrach:
	def load_csv(self,filename):
		dataset = list()
		with open(filename, 'r') as file:
			csv_reader = reader(file)
			for row in csv_reader:
				if not row:
					continue
				dataset.append(row)
		return dataset

	def str_column_to_float(self,dataset, column):
		for row in dataset:
			row[column] = float(row[column].strip())


	def rmse_metric(self,actual, predicted):
		sum_error = 0.0
		for i in range(len(actual)):
			prediction_error = predicted[i] - actual[i]
			sum_error += (prediction_error ** 2)
		mean_error = sum_error / float(len(actual))
		return sqrt(mean_error)


	def evaluate_algorithm(self,dataset, algorithm):
		test_set = list()
		for row in dataset:
			row_copy = list(row)
			row_copy[-1] = None
			test_set.append(row_copy)
		predicted = algorithm(dataset, test_set)
		print(predicted)
		actual = [row[-1] for row in dataset]
		rmse=sqrt(mean_squared_error(actual,predicted))
		print(rmse)

	def mean(self,values):
		return sum(values) / float(len(values))


	def covariance(self,x, mean_x, y, mean_y):
		covar = 0.0
		for i in range(len(x)):
			covar += (x[i] - mean_x) * (y[i] - mean_y)
		return covar


	def variance(self,values, mean):
		return sum([(x-mean)**2 for x in values])


	def coefficients(self,dataset):
		x = [row[0] for row in dataset]
		y = [row[1] for row in dataset]
		x_mean, y_mean = self.mean(x), self.mean(y)
		b1 = self.covariance(x, x_mean, y, y_mean) / self.variance(x, x_mean)
		b0 = y_mean - b1 * x_mean
		print('y=%f*x+%f' %(b1,b0))
		return [b0, b1]


	def simple_linear_regression(self,train, test):
		predictions = list()
		b0, b1 = self.coefficients(train)
		for row in test:
			yhat = b0 + b1 * row[0]
			predictions.append(yhat)
		return predictions

	def data(self):
		filename = 'LinearR_dataset.csv'
		dataset = self.load_csv(filename)
		for i in range(len(dataset[0])):
			self.str_column_to_float(dataset, i)

		self.evaluate_algorithm(dataset, self.simple_linear_regression)



class Usinglyb:
	def data(self):
		dataset=pd.read_csv("LinearR_dataset.csv")
		x=dataset.iloc[:,:-1].values
		y=dataset.iloc[:,1].values
		return x,y
	def model(self,x,y):
		from sklearn.linear_model import LinearRegression
		regressor=LinearRegression()
		regressor.fit(x,y)
		print("Accuracy: " ,regressor.score(x,y)*100)
		y_pred=regressor.predict([[10]])
		print(y_pred)


		hours=int(input('Enter the no of Hrs : '))
		eq=regressor.coef_*hours+regressor.intercept_
		print('y=%f*%f+%f' %(regressor.coef_,hours,regressor.intercept_))
		plt.plot(x,y,'^')
		plt.plot(x,regressor.predict(x))
		plt.show()

print('___________________From scratch_________________')
f1=Fromscrach()
f1.data()

print('___________________Using Lyb Function_________________')
f2=Usinglyb()
x,y=f2.data()
f2.model(x,y)


# In[ ]:




