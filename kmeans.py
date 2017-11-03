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

f = open(os.path.join(log_path, 'naive_bayes.log'), 'w')
def log(txt, do_print = 1):
	# if !isinstance(txt,str):
	# 	txt = str(txt)
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')



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

class kmeans():

    def __init__(self,dataset,num_clusters):
        self.num_clusters = num_clusters
        self.dataset = dataset
        self.dataset_without_labels = dataset[:,0:-1]
        # print np.shape(self.dataset_without_labels)
        # exit(0)
        self.num_features = self.dataset_without_labels.shape[1]
        self.initialiaze_centroids()
        self.do_kmeans()
        self.calculate_accuracy()
    
    def initialiaze_centroids(self):

        centroids = []
        for i in range(self.num_clusters) :
            centroids.append(self.dataset_without_labels[np.random.randint(0,len(self.dataset_without_labels))])
        self.centroids = np.array(centroids)
        print self.centroids
        # print self.get_closest_centroid(self.centroids[0],self.centroids)
    
    def get_closest_centroid(self,a):
        
        x = np.square(a-self.centroids)
        # print x
        # print np.sum(x,axis=1)
        return np.argmin(np.sum(x,axis=1),axis=0)
    
   


    
    def do_kmeans(self,max_iterations = 100):

        for i in range(max_iterations):
            old_centroid = self.centroids
            print 'iteration ',i
            partition = []
            new_centroids = []
            for i in range(self.num_clusters):
                partition.append([])
            
            for data_point in self.dataset:

                partition[self.get_closest_centroid(data_point[0:-1])].append(list(data_point))
            
            for cluster in partition :

                x = np.array(cluster)
                print 'cluster size ',len(cluster)
                # # print x
                # print cluster
                # exit(0)
                new_centroids.append(np.mean(x[:,0:-1],axis=0))
                
            
            self.centroids = np.array(new_centroids)
            self.partition = partition
            print self.centroids
            if(np.array_equal(old_centroid,self.centroids)):
                # print self.partition
                break
        # print self.centroids


    def calculate_accuracy(self):

        ans = []
        #works only when there are 2 classes, or else have to check all combinations

        for cluster in self.partition :

            x = np.array(cluster)
            ans.append(np.sum(x[:,-1]==0))
            ans.append(np.sum(x[:,-1]==1))
        print ans
        self.accuracy =  float(max((ans[0]+ans[3]),(ans[1]+ans[2])))/len(self.dataset)*100
                


##################################################################

if __name__ == "__main__":

	log ('soft_computing')
	np.random.seed(111)
	filename = 'spect.csv'
	data = load_csv(filename,'Yes')
	model = kmeans(data,2)
 	print 'Accuracy : ',model.accuracy


	# calculate_mean_std(data)
# 	total_accuracy=total_precision=total_recall=0.0
# 	for foldNumber in range(0,10):
# 		trainSet,testSet = splitDataset(data,foldNumber)
# 		accuracy,precision,recall = train_test(trainSet,testSet)
# 		total_accuracy+=accuracy
# 		total_precision+=precision
# 		total_recall+=recall

# 		log('foldNumber :'+str(foldNumber)+' |accuracy :'+str(accuracy)+' |precision :'+str(precision)+' |recall :'+str(recall))

# 	total_accuracy/=10.0
# 	total_precision/=10.0
# 	total_recall/=10.0
# 	log('AVERAGE :'+' |accuracy :'+str(total_accuracy)+' |precision :'+str(total_precision)+' |recall :'+str(total_recall))
f.close()