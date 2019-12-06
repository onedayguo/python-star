
from numpy import *
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
import os
import warnings
warnings.filterwarnings('ignore')

#--------------------------------load data---------------------------------

def loadDataLabel(file):
	f=open(file,"r")
	X=[]
	for l in f.readlines():
		_X=[]
		for i in range(len(l)):
			if l[i].isdigit():
				_X.append(int(l[i]))
		X.append(_X)
	return mat(X)

def loadDataFeature(file):
	f=open(file,"r")
	X=[]
	for l in f.readlines():
		l=l.split('\t')
		#print(l)
		_X=[]
		for i in range(len(l)):
			_X.append(int(l[i]))
		X.append(_X)
	return mat(X)


#--------------------------------save data---------------------------------

def saveData(X,file):
	m,n=X.shape
	f=open(file,'w+')
	for i in range(m):
		for j in range(n):
			f.write(str(X[i,j]))
			f.write(' ')
		f.write('\n')
	f.close()


#--------------------------------graph processing---------------------------


def graphSharpen(X):
	m,n=X.shape
	for i in range(m):
		for j in range(n):
			if X[i,j]<150:
				X[i,j]=0



#--------------------------------main function-----------------------------
if __name__ =="__main__":
	fx='data/digits4000_digits_vec.txt'
	fy='data/digits4000_digits_labels.txt'
	feature=loadDataFeature(fx)
	label=loadDataLabel(fy)
	graphSharpen(feature)
	'''
	for i in range(20):
		for j in range(20):
			print('%d'%label[i*28+j,0],end=' ')
		print('')
	'''
	X_train, X_test, y_train, y_test = train_test_split(feature,label,test_size=0.2, random_state=11)
	f='data/train_feature.txt'
	saveData(X_train,f)
	f='data/train_label.txt'
	saveData(y_train,f)
	f='data/test_feature.txt'
	saveData(X_test,f)
	f='data/test_label.txt'
	saveData(y_test,f)


	fx='data/cdigits_digits_vec.txt'
	fy='data/cdigits_digits_labels.txt'
	feature=loadDataFeature(fx)
	label=loadDataLabel(fy)
	#print(feature.shape)
	#print(label.shape)
	graphSharpen(feature)

	f='data/test_feature-c.txt'
	saveData(feature,f)
	f='data/test_label-c.txt'
	saveData(label,f)