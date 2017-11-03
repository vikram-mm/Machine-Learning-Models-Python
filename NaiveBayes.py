'''
@author: mmv

'''
import datetime
import csv
import os
import pandas as pd
import numpy as np
#from single_perceptron import splitDataset

#################################################################

log_path = '.'
# if not os.path.exists(path):
#         os.makedirs(path)

f1 = open(os.path.join(log_path, 'naive_bayes.log'), 'w')
def log(txt, do_print = 1):
	# if !isinstance(txt,str):
	# 	txt = str(txt)
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f1.write(txt + '\n')



#################################################################

def load_csv(filename,key=None):


	data_frame = pd.read_csv(filename)
	data = np.asarray(data_frame)

	if(key!=None):
		for i in range (0,data.size//data[0].size):
			#print(i)
			data[i][-1] = (data[i][-1]==key)

	np.random.shuffle(data)
	data = data.astype(np.float32)
	return data

#################################################################

def calculate_mean_std (data):

    mean = [[],[]] #mean[class,attribute]
    std = [[],[]]

    positive_data = data[data[:,-1]==1,:]
    #print positive_data.d
    positive_mean =  np.mean(positive_data,axis=0)
    positive_std = np.std(positive_data,axis=0)


    negative_data =  data[data[:,-1]==0,:]
    negative_mean =  np.mean(negative_data,axis=0)
    negative_std = np.std(negative_data,axis=0)
    # print negative_data


    # print positive_mean
    # print
    # print negative_mean
    # print
    # print positive_std
    # print
    # print negative_std

    return positive_mean,negative_mean,positive_std,negative_std

def splitDataset(dataset, foldNumber):
	trainSet = []
	testSet = []


	for i in range(0,len(dataset)):
		if(i%10==foldNumber):
			testSet.append(dataset[i])
		else:
			trainSet.append(dataset[i])

	return np.asarray(trainSet),np.asarray(testSet)

#################################################################

def pdf(x,mean,std):

    denom = np.sqrt(2*np.pi*std*std)
    num = np.exp(-np.square(x-mean)/(2*std*std))
    return num/denom


def train_test(trainSet,testSet):

    positive_mean,negative_mean,positive_std,negative_std = calculate_mean_std(trainSet)
    num_positive = positive_mean.shape[0]
    num_negative = negative_mean.shape[0]
    # prob_positive = num_positive * np.prod(pdf(testSet[0,0:-1],positive_mean[0:-1],positive_std[0:-1]))
    tp=tn=fp=fn=0.0
    for j in range(len(testSet)):

        positive_att_prob = pdf(testSet[j,0:-1],positive_mean[0:-1],positive_std[0:-1])
        negative_att_prob = pdf(testSet[j,0:-1],negative_mean[0:-1],negative_std[0:-1])
        # print testSet[0,0:-1]
        # print positive_mean[0:-1]
        # print positive_std[0:-1]
        # prob_positive = prob_positive.astype(np.float32)
        prob1 = num_positive
        prob2 = num_negative

        for i in range(len(positive_att_prob)):
            prob1 *= positive_att_prob[i]
            prob2 *= negative_att_prob[i]

        predicted_value = prob1>prob2
        # print predicted_value,testSet[j][-1]

        if(predicted_value == testSet[j][-1]):
            #print 'right'
              if(predicted_value==1.0):
                    tp+=1
              else:
                    tn+=1
        else:
            #print 'wrong'
          if(predicted_value==1.0):
                fp+=1
          else:
                fn+=1
        # print tp,tn,fp,fn
    #exit(0)
    # print '************************',len(testSet),tp,tn,fp,fn
    accuracy = (tp+tn)/(tp+tn+fp+fn)*100
    try:
              precision = tp/(tp+fp)*100
    except:
              precision = 0.0

    try:
              recall = tp/(tp+fn)*100
    except:
              recall = 0.0

    # print accuracy,precision,recall
    return accuracy,precision,recall

###############################################################

def objective_function(data):

    	total_accuracy=total_precision=total_recall=0.0
    	for foldNumber in range(0,10):
    		trainSet,testSet = splitDataset(data,foldNumber)
    		accuracy,precision,recall = train_test(trainSet,testSet)
    		total_accuracy+=accuracy
    		total_precision+=precision
    		total_recall+=recall

    		#log('foldNumber :'+str(foldNumber)+' |accuracy :'+str(accuracy)+' |precision :'+str(precision)+' |recall :'+str(recall))

    	total_accuracy/=10.0
    	total_precision/=10.0
    	total_recall/=10.0

        return total_accuracy,total_precision,total_recall



##################################################################

if __name__ == "__main__":

	log ('soft_computing')
	np.random.seed(64)
	filename = 'spect.csv'
	data = load_csv(filename,'Yes')

	# calculate_mean_std(data)
	total_accuracy=total_precision=total_recall=0.0
	for foldNumber in range(0,10):
		trainSet,testSet = splitDataset(data,foldNumber)
		accuracy,precision,recall = train_test(trainSet,testSet)
		total_accuracy+=accuracy
		total_precision+=precision
		total_recall+=recall

		log('foldNumber :'+str(foldNumber)+' |accuracy :'+str(accuracy)+' |precision :'+str(precision)+' |recall :'+str(recall))

	total_accuracy/=10.0
	total_precision/=10.0
	total_recall/=10.0
	log('AVERAGE :'+' |accuracy :'+str(total_accuracy)+' |precision :'+str(total_precision)+' |recall :'+str(total_recall))
f1.close()
