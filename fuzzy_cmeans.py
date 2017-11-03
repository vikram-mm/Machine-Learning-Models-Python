'''
@author: mmv

'''
import datetime
import csv
import os
import pandas as pd
import numpy as np

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

class cmeans():

    def __init__(self,dataset,num_clusters,m=2.0):
        self.num_clusters = num_clusters
        self.dataset = dataset
        self.dataset_without_labels = dataset[:,0:-1]
        self.m = m
        # print np.shape(self.dataset_without_labels)
        # exit(0)
        self.num_features = self.dataset_without_labels.shape[1]
        self.initialiaze_centroids()
        self.do_cmeans()
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
    
    
    def euclidean(self,a,b,feature_dim=2):

        return np.sqrt(np.sum(np.power(a-b,2),axis=feature_dim))


    
    def do_cmeans(self,max_iterations = 10000):

        for i in range(max_iterations):
            
            print "itertation : ",i
            # print self.dataset_without_labels.shape
            # print self.centroids.shape

            #computing membership matrix
            shape1 = self.dataset_without_labels.shape
            x = self.dataset_without_labels.reshape(1,shape1[0],shape1[1])

            shape2 = self.centroids.shape
            y = self.centroids.reshape(shape2[0],1,shape2[1])

            numer = self.euclidean(x,y)
            numer = np.power(numer,2.0/(self.m-1))

            # print numer.shape
            denom = np.sum(1.0/numer,axis=0,keepdims=True)
            # print denom.shape

            ans = 1.0/(numer*denom)
            # ans = 1.0/np.power(ans,2.0/(self.m-1))
            ans[np.isnan(ans)] = 1.0
            self.membership_matrix  = ans
            # print ans.shape
            # print ans#membership matrix
            # print np.sum(ans,axis=0)


            #calculating centroid
            numer2 = np.dot(np.power(ans,self.m),self.dataset_without_labels)
            denom2 = np.sum(np.power(ans,self.m),axis=1,keepdims=True)
            new_centroids = numer2/denom2
            print new_centroids
            if(np.array_equal(np.around(new_centroids,3),np.around(self.centroids,3))):
                break
            self.centroids = new_centroids
            


    




    def calculate_accuracy(self):

        ans = []
        #works only when there are 2 classes, or else have to check all combinations
        new_centroids = []
        partition = []
        for i in range(self.num_clusters):
            partition.append([])
        
        for data_point in self.dataset:

            partition[self.get_closest_centroid(data_point[0:-1])].append(list(data_point))
        for cluster in partition :

            x = np.array(cluster)
            ans.append(np.sum(x[:,-1]==0))
            ans.append(np.sum(x[:,-1]==1))
        print 'split: ',ans
        self.accuracy =  float(max((ans[0]+ans[3]),(ans[1]+ans[2])))/len(self.dataset)*100
                


##################################################################

if __name__ == "__main__":

	log ('soft_computing')
	np.random.seed(10)
	filename = 'spect.csv'
	data = load_csv(filename,'Yes')
	model = cmeans(data,2)
	print 'membership matrix :'
	print model.membership_matrix
 	print 'Accuracy : ',model.accuracy


f.close()