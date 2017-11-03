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

f = open(os.path.join(log_path, 'spect_detail.log'), 'w')
f1 = open(os.path.join(log_path, 'spect.log'), 'w')
def log(txt, do_print = 1, p2 = 0):
	# if !isinstance(txt,str):
	# 	txt = str(txt)
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')
    if(p2==1):
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

	return data

def sigmoid(x):
   return .5 * (1 + np.tanh(.5 * x))
	#return max(0,z)

def desigmoid(z):

	return sigmoid(z)*(1-sigmoid(z))
	# if(z>0):
	# 	return 1.0
	# else:
	# 	return 0.0

def splitDataset(dataset, foldNumber):
	trainSet = []
	testSet = []


	for i in range(0,len(dataset)):
		if(i%10==foldNumber):
			testSet.append(dataset[i])
		else:
			trainSet.append(dataset[i])

	return np.asarray(trainSet),np.asarray(testSet)

class mlp_single_hidden_layer():

    def __init__(self,num_units,num_features,num_classes=2):

        self.num_units = num_units #in hidden layer
        #self.LR = LR
        self.num_classes = num_classes
        self.num_features = num_features
        self.initialize_wb()


    def initialize_wb(self):

        b = np.float32(1.0)
        self.weights_hidden = np.random.normal(1.0/(self.num_features*self.num_units),0.05,(self.num_units,self.num_features))
        #self.weights_hidden = np.random.normal(0.0,0.05,(self.num_units,self.num_features))
	#print self.weights_hidden.shape
        self.weights_output = np.random.normal(1.0/(self.num_units),0.05,(self.num_classes,self.num_units))
        #self.weights_output = np.random.normal(0.0,0.05,(self.num_classes,self.num_units))
	#print self.weights_output.shape
        # self.bias_hidden = np.repeat(self.num_units,b)
        self.bias_hidden = np.random.normal(1.0/(self.num_features*self.num_units),0.05,(self.num_units))
        self.bias_hidden = self.bias_hidden.astype(np.float64)

        #self.bias_output = np.repeat(self.num_classes,b)
        self.bias_output = np.random.normal(1.0/(self.num_features*self.num_units),0.05,(self.num_classes))
        self.bias_output = self.bias_output.astype(np.float64)
        # print self.bias_hidden.shape
        # exit(0)
    def train(self,trainSet,LR=0.1,max_iterations=500):

		for j in range(max_iterations):

			for i in range(len(trainSet)):
				self.forward_prop(trainSet,i)
				self.back_prop(trainSet,LR,i)
				#print '**************************'

    def forward_prop(self,trainSet,i):

		self.hidden_nodes = np.dot(self.weights_hidden,trainSet[i][0:-1]) #+ self.bias_hidden
		self.hidden_nodes = self.hidden_nodes.astype (np.float32)
	        #print self.hidden_nodes.shape
		#print self.hidden_nodes.dtype
		self.hidden_activation = sigmoid(self.hidden_nodes)
		self.output_nodes = np.dot(self.weights_output,self.hidden_activation) #+ self.bias_output
		self.output_activation = sigmoid(self.output_nodes)

    def back_prop(self,trainSet,LR,i):

		#print 'weights before' , '\n',self.weights_output ,'\n',self.weights_hidden
		one_hot = np.arange(2)==trainSet[i][-1]
		#print 'one_hot : ',one_hot
		#print 'self.output_nodes' , self.output_nodes , np.arange(2)==trainSet[i][-1]
		output_error = (self.output_activation - one_hot) * desigmoid(self.output_nodes)
		#print 'output_error' , output_error
		hidden_error = np.dot(self.weights_output.T,output_error) * desigmoid(self.hidden_nodes)

		delta_weight_output = np.dot(np.expand_dims(output_error,axis=1),np.expand_dims(self.hidden_activation,axis=0))
		#delta_weight_output = np.dot(output_error,self.hidden_nodes.T)
		delta_weight_output = delta_weight_output.astype(np.float32)
		delta_weight_hidden = np.dot(np.expand_dims(hidden_error,axis=1),np.expand_dims(trainSet[i][0:-1],axis=0))

		delta_weight_hidden = delta_weight_hidden.astype(np.float32)
		self.weights_output -= delta_weight_output * LR
		self.weights_hidden -= delta_weight_hidden * LR

		# self.bias_output -= output_error * LR
		# self.bias_hidden -=  hidden_error * LR
		#print 'weights after' , '\n',self.weights_output ,'\n',self.weights_hidden
		#print self.test(trainSet[i:i+1])

    def test(self,testSet):

		#print 'weights test' , '\n',self.weights_output ,'\n',self.weights_hidden
		tp=tn=fp=fn=0.0

		for i in range(len(testSet)):
			self.forward_prop(testSet,i)
			#log(str(self.output_activation))
			#log(str(np.argmax(self.output_activation)) +' '+ str(testSet[i][-1]))
			predicted_value = np.argmax(self.output_activation)

			#print 'hi'
			if(predicted_value == testSet[i][-1]):
    			#print 'right'
    			  if(predicted_value==0.0):
        				tp+=1
    			  else:
        				tn+=1
    			else:
        			#print 'wrong'
    			  if(predicted_value==0.0):
        				fp+=1
    			  else:
        				fn+=1
		#print tp,tn,fp,fn
		#exit(0)
		accuracy = (tp+tn)/(tp+tn+fp+fn)*100
		try:
		          precision = tp/(tp+fp)*100
		except:
                  precision = 0.0

		try:
		          recall = tp/(tp+fn)*100

		except:
		          
		          recall = 0.0
		          

		
		return accuracy,precision,recall



if __name__ == "__main__":

	log ('soft_computing, spectf dataset, num_iterations = 500')
	np.random.seed(10)
	filename = 'spect.csv'
	data = load_csv(filename,'Yes')

	# filename = 'IRIS.csv'
	# data = load_csv(filename,'Iris-setosa')
	num_features = data[0].size-1
	# print data[:,-1]
	# exit(0)

	

	LR = 0.1
	while(LR<=1.0):

		total_accuracy=total_precision=total_recall=0.0
		for foldNumber in range(0,10):
			model = mlp_single_hidden_layer(5,num_features)
			trainSet,testSet = splitDataset(data,foldNumber)
			model.train(trainSet,LR)
			accuracy,precision,recall = model.test(testSet)
			total_accuracy+=accuracy
			total_precision+=precision
			total_recall+=recall

			log('LR :'+str(LR)+' |foldNumber :'+str(foldNumber)+' |accuracy :'+str(format(accuracy,'0.2f'))+' |precision :'\
				+str(format(precision,'0.2f'))+' |recall :'+str(format(recall,'0.2f')) )

		total_accuracy/=10.0
		total_precision/=10.0
		total_recall/=10.0
		log('AVERAGE : '+'LR :'+str(LR)+' |accuracy :'+str(format(total_accuracy,'0.2f'))+' |precision :'\
				+str(format(total_precision,'0.2f'))+' |recall :'+str(format(total_recall,'0.2f')),p2=1)
		log('******************************************************************')
		LR += 0.1

f.close()
f1.close()
