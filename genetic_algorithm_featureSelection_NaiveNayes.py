'''
@author: mmv

'''
import datetime
import csv
import os
import pandas as pd
import numpy as np
from naive_bayes import objective_function
#from single_perceptron import splitDataset

#################################################################

log_path = '.'
# if not os.path.exists(path):
#         os.makedirs(path)

f = open(os.path.join(log_path, 'ga.log'), 'w')
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

################################################################

class ga():

    def __init__(self,objective_function,data,max_chromosomes=31,min_chromosomes=30,cross_over_ratio=0.25,mutation_ratio=0.1):

        self.data = data
        self.num_features = data.shape[1] - 1
        self.max_chromosomes = max_chromosomes
        self.min_chromosomes = min_chromosomes
        self.cross_over_ratio = cross_over_ratio
        self.mutation_ratio = mutation_ratio
        self.best_accuracy = 0
        self.initialize_population()
        self.do_ga()

    def do_ga(self,num_iterations=100):

		
        for i in range(num_iterations):
            self.selection()
            self.cross_over()
            self.mutation()
            max_ans = self.fitness_function(self.population[0])
            good_chromosome = self.population[0]

            for chromosome in self.population:
                k = self.fitness_function(chromosome)
                if(k > max_ans):
                    max_ans = k
                    good_chromosome = chromosome

            log('iteration '+str(i)+' : highest among all chromosomes '+str(max_ans))
            if(max_ans>self.best_accuracy):
            	self.best_accuracy = max_ans
            	self.best_chromosome = good_chromosome



    def initialize_population(self):

        self.num_chromosomes = np.random.randint(self.min_chromosomes,self.max_chromosomes)
        log('num_chromosomes '+str(self.num_chromosomes))
        self.population = np.random.randint(2,size=(self.num_chromosomes,self.num_features))
        # self.population[:,-1] = 1
        # print self.poluation
        # print self.poluation.shape

    def selection(self):

        # print self.data[self.poluation[0]].shape
        # print np.sum(self.population[0])
        # print data[:,self.population[0]==1]
        # print data[:,self.population[0]==1].shape
        fitness_values = []

        for chromosome in self.population:
            # print self.fitness_function(chromosome)
            fitness_values.append(self.fitness_function(chromosome))

        fitness_values = np.asarray(fitness_values)
        total_fitness = np.sum(fitness_values)
        cummulative_probaility = [fitness_values[0]/total_fitness]
        prev_cumulative = cummulative_probaility[0]

        for f in fitness_values[1:]:
            next_cumulative = f/total_fitness + prev_cumulative
            prev_cumulative = next_cumulative
            cummulative_probaility.append(next_cumulative)

        # print cummulative_probaility

        duplicate_population =  np.array(self.population)

        # print 'duplicate_population',duplicate_population
        for i in  range(len(self.population)):
            rand_num = np.random.random_sample()
            # print rand_num
            selected_chromosome_index = np.searchsorted(cummulative_probaility,rand_num)
            # print selected_chromosome_index
            # print 'before',self.population[i]
            self.population[i] = duplicate_population[selected_chromosome_index]
            # print 'after',self.population[i]

        # print 'duplicate_population',duplicate_population
        # print 'population',self.population



        # for i in range(ceil(cross_over_ratio*))

    def cross_over(self):

        num_cross_overs = np.ceil(self.num_chromosomes*self.cross_over_ratio)
        num_cross_overs = int(num_cross_overs)
        # print num_cross_overs
        # print type(num_cross_overs)
        if(num_cross_overs%2==1):
            num_cross_overs+=1
        # print num_cross_overs

        cross_over_chromosome_indices = np.random.randint(self.num_chromosomes,size=(num_cross_overs))

        # print cross_over_chromosome_indices

        # cross_over_chromosomes = self.population[cross_over_chromosome_indices]
        # print cross_over_chromosomes

        i=0

        while(i<num_cross_overs):

            c1 = cross_over_chromosome_indices[i]
            c2 = cross_over_chromosome_indices[i+1]
            cross_over_point = np.random.randint(0,self.num_features)
            # print '****************************************'
            # print cross_over_chromosomes[i],cross_over_chromosomes[i+1]
            # print 'cross_over_point', cross_over_point
            temp = np.array(self.population[c1,0:cross_over_point])
            self.population[c1,0:cross_over_point] = self.population[c2,0:cross_over_point]
            self.population[c2,0:cross_over_point] = temp
            # print cross_over_chromosomes[i],cross_over_chromosomes[i+1]
            # print '****************************************'
            i+=2

        # print self.population

    def mutation(self):

        num_mutations = np.ceil(self.num_chromosomes*self.mutation_ratio)
        num_mutations = int(num_mutations)
        mutation_indices = np.random.randint(self.num_chromosomes,size=(num_mutations))
        # print mutation_indices

        for i in mutation_indices:

            # print '*********'
            # print self.population[i]
            rand_bit = np.random.randint(0,self.num_features)
            # print rand_bit
            self.population[i,rand_bit] = 1 - self.population[i,rand_bit]

            # print self.population[i]
            # print '*********'

    def fitness_function(self,chromosome,give_all=False):

        # print self.data[:,chromosome==1].shape
        # print self.data[:,-1:0].shape
        labels = np.expand_dims(self.data[:,-1],1)
        test_data = np.concatenate((self.data[:,0:-1][:,chromosome==1],labels),axis=1)
        # print 'test_shape',test_data.shape
        # print 'obj ******',objective_function(test_data)
        if(give_all==False):
        	return objective_function(test_data)[0]
        else:
        	return objective_function(test_data)

#################################################################

if __name__ == "__main__":

    log ('soft_computing')
    np.random.seed(111)
    filename = 'spect.csv'
    data = load_csv(filename,'Yes')

    log('initial'+str(objective_function(data)))
    model = ga(objective_function,data)

    log('best_chromosome : '+ str(model.best_chromosome))
    print 'sum :' ,np.sum(model.best_chromosome)
    log('best accuracy : '+str(model.best_accuracy))
    _,precision,recall = model.fitness_function(model.best_chromosome,True)
    log('precision : '+str(precision))
    log('recall : '+str(recall))
f.close()

