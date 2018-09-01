import numpy as np # linear algebra
from numpy.core.umath_tests import inner1d
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import CategoricalDtype
from sklearn import model_selection, ensemble, metrics, linear_model
from sklearn.model_selection import *
from sklearn.preprocessing import *
import matplotlib.pyplot as plt
import os
import math


def predict(theta,X):
	X = np.insert(X,0,1,axis=1)
	y = hypothesis(theta,X)
	X = np.delete(X,0,axis=1)
	return y

def hypothesis(theta,X):
	mul = np.multiply(X,theta)
	h = np.sum(mul,axis=1)
	return h

def RMSE(theta,X,y):
	y_linha = hypothesis(theta,X)
	error = math.sqrt(((y_linha-y)**2).mean())
	return error

# 
#  This method performs the Batch Gradient Descent to optimize a
# cost function and define the best parameters for the linear 
# regression model
#
# params
#   X          -> list of features
#   y          -> list of targets
#   alpha      -> learning rate
#   iterations -> max number of iterations
#
# return
#   theta -> the set with the best parameters
#   error -> minimum error found
#
def BGD(X,y, alpha, iterations):

	X = np.insert(X,0,1,axis=1)
	print(X)
	nsamples = X.shape[0]
	nfeatures = X.shape[1]
	theta = np.zeros(nfeatures)
	thetaError = []

	for i in range(iterations):

		# computing hypothesis H(x)
		h = hypothesis(theta,X)

		# computing error
		error = h - y

		# computing gradient
		grad = np.dot(X.transpose(),error)/nsamples

		# applying gradinet descent
		theta = theta - alpha*grad

		# compute the error of the current theta
		thetaError.append(RMSE(theta,X,y))
		print("epoch: ",str(i),end="")
		print("    RMSE error: ",str(thetaError[i]))


	X = np.delete(X,0,axis=1)

	plt.plot(thetaError)
	plt.ylabel('Error')
	plt.xlabel('iterations')
	plt.show()

	return theta,thetaError[iterations-1]


# 
#  This method performs the Stocastic Gradient Descent to optimize a
# cost function and define the best parameters for the linear 
# regression model
#
# params
#   X          -> list of features
#   y          -> list of targets
#   alpha      -> learning rate
#   iterations -> max number of iterations
#
# return
#   theta -> the set with the best parameters
#   error -> minimum error found
#
def SGD(X,y, alpha, iterations):

	X = np.insert(X,0,1,axis=1)
	#print(X)
	nsamples = X.shape[0]
	nfeatures = X.shape[1]
	theta = np.zeros(nfeatures)
	thetaError = []

	for i in range(iterations):

		for j in range(nsamples):

			#random_index = np.random.randint(nsamples)
			random_index = j
			#print(random_index)
			#print("y: ",str(y[j]))

			X_sample = X[random_index]
			y_sample = y[random_index]

			# computing hypothesis H(x)			
			h = np.dot(X_sample,theta)

			# computing error
			error = h - y_sample

			# computing gradient
			grad = X_sample*error

			'''		
			if (math.isnan(error)):				
				print(X_sample)
				print(y_sample)
				print(h)
				print(error)
				return
			'''

			# applying gradinet descent
			theta = theta - alpha*grad

		# compute the error of the current theta
		thetaError.append(RMSE(theta,X,y))
		#print("epoch: ",str(i),end="")
		#print("    RMSE error: ",str(thetaError[i]))

	X = np.delete(X,0,axis=1)

	
	'''
	plt.plot(thetaError)
	plt.ylabel('Error')
	plt.xlabel('iterations')
	plt.show()
	'''
	
	

	return theta,thetaError[iterations-1]



#  For linear regression, it is possible to estimate the values
# of all parameters theta by applying the normal equation method,
# which corresponds to the following equation:
#
#  Theta = (Xt.X)^-1.Xt.y
#
#  This procedure is called Normal Equation, which is implemented
# here
#
# params:
#   X -> set of features
#   Y -> set of targets
#
# return:
#   theta -> set of parameters
#
def normalEquation(X,y):
	X = np.insert(X,0,1,axis=1)
	npX = np.copy(X)
	npY = y.transpose()
	npXt = npX.transpose()

	R1 = np.matmul(npXt,npX)

	det = np.linalg.det(R1)

	if (det != 0):
		R1 = np.linalg.inv(R1)
		R2 = np.matmul(npXt,npY)
		theta = np.matmul(R1,R2)
	else:
		theta = []
		print("Error! Matrix (Xt.X) has no inverse.")

	error = RMSE(theta,X,y)
	print("Normal Equation --- RMSE error: ",str(error))

	X = np.delete(X,0,axis=1)

	return theta


#
#  This method separates the K-fold into train and test folds
#  
# params:   
#   folds -> folds containing the training sets divided. Shape: (k,nsamples/k,nfeatures)
#   i     -> index of the fold to be used as test
#
# return:
#   fold_train -> fold of training samples
#   fold_test  -> fold of test samples
#
def separateTrainTestFolds(folds,i):
	fold_train = np.concatenate([folds[:i],folds[i+1:]])
	fold_test = folds[i]
	fold_X_train = fold_train.reshape((fold_train.shape[1]*fold_train.shape[0],fold_train.shape[2]))
	fold_y_train = fold_X_train[:,fold_X_train.shape[1]-1]
	fold_X_test = fold_test.copy()
	fold_y_test = fold_X_test[:,fold_X_test.shape[1]-1]
	return fold_X_train, fold_y_train, fold_X_test, fold_y_test


#
#  This method performs a k-fold cross validation in order to define the
# best hyperparameters for the linear regression model
#  
# params:
#   X_train       -> training features
#   y_train       -> training target
#   nfolds        -> number of foldes for the k-fold method
#   learning_rate -> set of learning rates candidates (list)
#   iterations    -> set of iterations candidates (list)
#
# return:
#   set of best parameters
#
def Kfold(X,y,k):

	nsamples = X.shape[0]
	nfeatures = X.shape[1]

	X = np.insert(X,nfeatures,y,axis=1)
	nfeatures += 1 # added the target in the set of features	

	np.random.shuffle(X)
	nfold = nsamples // k
	rest = nsamples % k

	X = X.reshape((k,nfold,nfeatures))

	if (rest != 0):
		Xrest = X[-rest:]
		X = X[:-rest]	
		j = 0
		for x in Xrest:
			X[j].append(x)
			j += 1

	return X

#
#  This method performs a k-fold cross validation in order to define the
# best hyperparameters for the linear regression model
#  
# params:
#   X_train       -> training features
#   y_train       -> training target
#   nfolds        -> number of foldes for the k-fold method
#   learning_rate -> set of learning rates candidates (list)
#   iterations    -> set of iterations candidates (list)
#
# return:
#   set of best parameters
#
def crossValidation(X_train,y_train,nfolds,learning_rate,iterations):

	X_bk = X_train.copy()
	y_bk = y_train.copy()
	folds = Kfold(X_bk,y_bk,nfolds)
	fold_X_train, fold_y_train, fold_X_test, fold_y_test = separateTrainTestFolds(folds,1)

	BGD(fold_X_train, fold_y_train, 0.001, 5000)

	#for f in range(nfolds):
	#	fold_X_train, fold_y_train, fold_X_test, fold_y_test = separateTrainTestFolds(folds,f)
	#	for alpha in learning_rate:
	#		for it in iterations:



base_dir = 'input/'
print(os.listdir(base_dir))

df_diamonds_train = pd.read_csv('%s/diamonds-train.csv'%(base_dir))
df_diamonds_train.head(10)

cuts_ordered = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
df_diamonds_train['cut'] = df_diamonds_train['cut'].astype(CategoricalDtype(cuts_ordered, ordered=True))

colors_ordered = [ 'J','I','H','G','F','E','D']
df_diamonds_train['color'] = df_diamonds_train['color'].astype(CategoricalDtype(colors_ordered, ordered=True))

clarity_codes = {'I3','I2','I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF','FL'}
df_diamonds_train['clarity'] = df_diamonds_train['clarity'].astype(CategoricalDtype(clarity_codes, ordered=True))

cat_columns = df_diamonds_train.select_dtypes(['category']).columns.values
df_diamonds_train[cat_columns] = df_diamonds_train[cat_columns].apply(lambda x: x.cat.codes)

df_diamonds_train = df_diamonds_train.drop(df_diamonds_train.loc[df_diamonds_train.x <= 0].index)
df_diamonds_train = df_diamonds_train.drop(df_diamonds_train.loc[df_diamonds_train.y <= 0].index)
df_diamonds_train = df_diamonds_train.drop(df_diamonds_train.loc[df_diamonds_train.z <= 0].index)


df_diamonds_train['volume'] = df_diamonds_train['x'] * df_diamonds_train['y'] * df_diamonds_train['z']
df_diamonds_train['ratioXY'] = df_diamonds_train['x'] / df_diamonds_train['y']
df_diamonds_train['ratioXZ'] = df_diamonds_train['x'] / df_diamonds_train['z']
df_diamonds_train.pop('x')
df_diamonds_train.pop('y')
df_diamonds_train.pop('z')

#train = df_diamonds_train.head(10)
train = df_diamonds_train

X_train = train.copy()
y_train = X_train.pop('price')

#print(X_train)
#print(y_train)

scaler = RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
#print(X_train)
#print(X_train.shape)

#crossValidation(X_train,y_train,5,[0.01,0.05,0.001],[10000,15000])
#BGD(X_train, y_train, 0.005, 10000)
SGD(X_train, y_train, 0.005, 5000)
normalEquation(X_train,y_train)