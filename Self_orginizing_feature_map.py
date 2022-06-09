###### -*- coding: utf-8 -*-
# """
# Created on Mon May  2 13:03:53 2022

# @author: bsa91
# reference https://stackabuse.com/self-organizing-maps-theory-and-implementation-in-python-with-numpy/
# """


from numba import cuda, float64
import numpy as np
from math import exp, pow
import matplotlib.pyplot as plt
class Sofm():
    #possible improvement add range for weights and option to normalize or not
    def __init__(self, x_dim, y_dim, weights, r_state = 0, normalize = True,
                 max_val = 255, negative = False):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.weights = weights
        if negative:
            lower_start = -max_val
        else:
            lower_start = 0
        #set random state, seed can be optionally given in declaration
        self.rand = np.random.RandomState(0)
        #set random weights for the sofm 
        self.sofm = self.rand.randint(lower_start, max_val, (x_dim, y_dim, weights)).astype(float)
        #normalize
        if normalize:
            self.sofm = self.sofm/10000000 
    
    #data should be an np array of the vectors so if only one sample [[x1, x2, ...]]
    #steps define the number of steps away from BMU for which nodes should be attemped 
    #to be adjusted
    def train(self, data, learn_rate = .1, radius_sq = 1, lr_decay = .1, 
              radius_decay = .1, epochs = 10, steps = 3, maxThreads =1024):
        #transfer the sofm to the device
        d_sofm=cuda.to_device(self.sofm)
        
        radius_sq_org = radius_sq
        learn_rate_org = learn_rate
        #check that the dimesions and type is correct
        if data.ndim != 2:
            raise Exception("incorrect data dimesions")
        elif data.dtype != "float":
            raise Exception("incorrect data type, expected float")
        elif data.shape[1] != self.weights:
            raise Exception("incorrect vector length for data")
        r_data = np.copy(data)
        #for loop for epochs should call kernel function for finding distances
        for i in range(epochs):
            #for loop for all data
            self.rand.shuffle(r_data)
            for x in r_data:
                d_data = cuda.to_device(x)
                d_distance =  cuda.to_device(np.zeros((self.x_dim, self.y_dim)))
                
                threadsperblock = (self.weights // 32 + 1) * 32 if self.weights < 1024 else 1024  
                blockpergrid = (self.x_dim, self.y_dim)             
                
                #find the distance between the nodes and the vector
                self.find_distances[blockpergrid, threadsperblock](d_sofm, d_data, d_distance)
              
                #finds the best matching unit, third spot for the distance
                #might be faster or equivalent to do on host
                #considerationis time to copy distances over to host from device 
                threadsperblock = (self.y_dim // 32 + 1) * 32 if self.y_dim < 1024 else 1024
                blockpergrid = self.x_dim                 
               
                d_bmu = cuda.to_device(np.zeros((blockpergrid, 3)))
                #revisit number of blocks and threads mabey
                self.find_bmu[blockpergrid, threadsperblock](d_distance, d_bmu)  
                
                #copies Best Matching unit back to host                
                bmu = d_bmu.copy_to_host()
                temp = bmu[np.argmin(bmu, axis = 0)[2]]
                bmu = temp
               
                #find neighborhood on host                  
                #kernel for updating neighborhood 
                if learn_rate < 1e-3:
                    steps = 0
                hyper_params = np.array([learn_rate, radius_sq, steps ])
                d_hyper_params = cuda.to_device(hyper_params)
                #give bottom left corner and kernels add to that for indexing, also includes the bmu in the second dim
                corn = np.array([[max(bmu[0] - steps, 0), 
                                  max(bmu[1] - steps, 0)],
                                 [bmu[0], bmu[1]]], dtype = int)
                d_corn = cuda.to_device(corn)
                
                #caculate the max between the edge of the array and the nodes to look at 
                maxx = min(self.x_dim - 1,  bmu[0] + steps)
                maxy = min(self.y_dim - 1,  bmu[1] + steps)            
                
                
                threadsperblock = (self.weights % 32 + 1) * 32 if self.weights < 1024 else 1024 
                #create blocks for the number of units that need to be updated
                blockpergrid = (int(maxx - corn[0][0] + 1), int(maxy - corn[0][1] + 1))

                self.update_neighborhood[blockpergrid, threadsperblock](d_sofm, d_data,
                                                                        d_hyper_params, d_corn)            
            
            
            #adjust learning rate and radius
            learn_rate = learn_rate_org * exp(-i * lr_decay)
            radius_sq = radius_sq_org * exp(-i * radius_decay)
        self.sofm = d_sofm.copy_to_host()  
    
    def return_nearest(self, data):
        
        if data.ndim != 1:
            raise Exception("wrong dimensions on data")
        elif data.dtype != "float":
            raise Exception("incorrect data type, expected float")
        elif data.shape[0] != self.weights:
            raise Exception("incorrect vector length for data")
    
        d_data = cuda.to_device(data)
        d_distance =  cuda.to_device(np.zeros((self.x_dim, self.y_dim)))
        
        d_sofm = cuda.to_device(self.sofm)
        
        threadsperblock = (self.weights // 32 + 1) * 32 if self.weights < 1024 else 1024  #todo  
        blockpergrid = (self.x_dim, self.y_dim)
                
                
              
                
                
        #find the distance between the nodes and the vector
        self.find_distances[blockpergrid, threadsperblock](d_sofm, d_data, d_distance)
              
        distance = d_distance.copy_to_host()
       
        return np.argmin(distance, axis = None)        
      
    
    
    #kernel method to find distances of data to each sofm node
    #stores the result on the device in d_distance 
    @cuda.jit
    def find_distances(d_sofm, d_data, d_distance):
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        tx = cuda.threadIdx.x
        if tx < d_data.shape[0] and bx < d_distance.shape[0] and by < d_distance.shape[1]:
            temp = pow(d_sofm[bx][by][tx] - d_data[tx], 2)
            cuda.atomic.add(d_distance, (bx, by), temp)
    #kernel function to find the best matching unit, to be stored in d_bmu, if more than 1024 returns multiple
    #and need to find best on host
    @cuda.jit
    def find_bmu(d_distance, d_bmu):
        tx = cuda.threadIdx.x
        dx = cuda.blockIdx.x 
        temp = cuda.shared.array(1, float64)
        
        if tx == 0:
            temp[0] = d_distance[dx][tx]
        
      
        
       
        if dx < d_distance.shape[0] and tx < d_distance.shape[1]:
            cuda.atomic.min(temp, 0, d_distance[dx][tx])
        
        cuda.syncthreads()
        
        if dx < d_distance.shape[0] and tx < d_distance.shape[1] and temp[0] == d_distance[dx][tx]:
            d_bmu[dx][0] = dx
            d_bmu[dx][1] = tx
            d_bmu[dx][2] = d_distance[dx][tx]
        
    
   
    #kernel function to update the neighborhood of bmu 
    #hyper params containg the learning rate at index 0, 
    #radius at index 1
    #and steps at radius
    #d_corn contains the cordinates of the bottom left corner in index 0
    #the bmu at index 1
    #possible future optimization have bmu coord and sofm unit coord be shared mabey faster
    #only on thread per block retrieves or possibly on host?
    @cuda.jit
    def update_neighborhood(d_sofm, d_data, d_hyper_params, d_corn):
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        tx = cuda.threadIdx.x
        
        bmu_x = d_corn[1][0]
        bmu_y = d_corn[1][1]
        
        #x and y coordinate of unit all threads in this block will deal with
        sofm_x = d_corn[0][0] + bx 
        sofm_y = d_corn[0][1] + by
        if tx < d_sofm.shape[2]:
            #weight specific to this thread
            sofm_weight = d_sofm[sofm_x][sofm_y][tx]
            
            #distance between bmu and unit this thread block deals with
            dist = pow(bmu_x - sofm_x, 2) + pow(bmu_y - sofm_y, 2)
            
            neighborhood_dist = exp(-dist / 2 / d_hyper_params[1])
            
            sofm_weight += d_hyper_params[0] * neighborhood_dist * (d_data[tx] - sofm_weight)
            
            d_sofm[sofm_x][sofm_y][tx] = sofm_weight
    
    #mainly for training purposes    
    def img_plot_sofm(self):
        plt.imshow(self.sofm.astype(int))
        
    def coord_plot_sofm(self):
        plt.plot(self.sofm.astype(int))
        


m = 10
n = 10
# Number of training examples
n_x = 3000
rand = np.random.RandomState(0)
# Initialize the training data
train_data = rand.randint(0, 255, (n_x, 3)).astype(float)

sofm = Sofm(10, 10, train_data.shape[1], normalize = False)
sofm.plot_sofm()
sofm.train(train_data)
sofm.plot_sofm()