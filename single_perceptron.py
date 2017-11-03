'''
@author: mmv

'''

import csv
import pandas as pd
import numpy as np


def load_csv(filename,key='none'):


	data_frame = pd.read_csv(filename)
	data = np.asarray(data_frame)

	if(key!='none'):
		for i in range (0,data.size//data[0].size):
			#print(i)
			data[i][-1] = (data[i][-1]==key)

	np.random.shuffle(data)

	return data




def train(train_data,LR,weights,bias,num_iterations=500):


	for j in range (num_iterations):
		
		for i in range (0,len(train_data)):

			predicted_value = np.sign(np.dot(train_data[i][0:-1],weights)+bias)

			predicted_value=np.clip(predicted_value,0,1)

			#print ('predicted value : ', predicted_value,' label : ',train_data[i][-1]+0.0)
			#print (data[i][-1])
			delta_weight = LR * (predicted_value- train_data[i][-1]) * train_data[i][0:-1]
			bias = bias - LR * (predicted_value- train_data[i][-1])
			#print (delta_weight)
			weights = weights - delta_weight

		accuracy,_,_ = test(train_data,weights,bias)

		if(accuracy==100.0):
				break


	return weights,bias

def test(test_data,weights,bias):

	tp=tn=fp=fn=0.0

	for i in range(0,len(test_data)):
		predicted_value = np.sign(np.dot(test_data[i][0:-1],weights)+bias)

		predicted_value=np.clip(predicted_value,0,1)

		if(predicted_value == test_data[i][-1]):

			if(predicted_value==1.0):
				tp+=1
			else:
				tn+=1
		else:

			if(predicted_value==1.0):
				fp+=1
			else:
				fn+=1


	accuracy = (tp+tn)/(tp+tn+fp+fn)*100
	try:
		precision = tp/(tp+fp)*100
	except:
		precision=0.0
	recall = tp/(tp+fn)*100

	return accuracy,precision,recall

def splitDataset(dataset, foldNumber):
	trainSet = []
	testSet = []


	for i in range(0,len(dataset)):
		if(i%10==foldNumber):
			testSet.append(dataset[i])
		else:
			trainSet.append(dataset[i])

	return np.asarray(trainSet),np.asarray(testSet)



if __name__ == "__main__":

	np.random.seed(11)
	# filename = 'IRIS.csv'
	# data = load_csv(filename,'Iris-setosa')
	filename = 'spect.csv'
	data = load_csv(filename,'Yes')

	#weight initialisation
	LR = 0.1
	while(LR<=1.0):
		weights = []

		num_features = data[0].size-1

		for i in range(0,num_features):
			weights.append(1.0/((num_features)+1))

		weights = np.asarray(weights)

		bias = np.float32(1.0)
		#LR = np.float32(0.1)

		total_accuracy=total_precision=total_recall=0.0


		for foldNumber in range(0,10):
			weights = []

			num_features = data[0].size-1

			for i in range(0,num_features):
				weights.append(1.0/((num_features)+1))

			weights = np.asarray(weights)

			bias = np.float32(1.0)
			trainSet,testSet = splitDataset(data,foldNumber)
			weights,bias=train(trainSet,LR,weights,bias)
			accuracy,precision,recall = test(testSet,weights,bias)

			print 'LR :'+str(LR)+' |foldNumber :',foldNumber,' |accuracy :',accuracy,' |precision :',precision,' |recall :',recall

			total_accuracy+=accuracy
			total_precision+=precision
			total_recall+=recall


		total_accuracy/=10.0
		total_precision/=10.0
		total_recall/=10.0
		print 'AVERAGE'+'accuracy :',total_accuracy,' |precision :',total_precision,' |recall :',total_recall
		LR += 0.1
