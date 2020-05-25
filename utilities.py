import numpy as np
import torch
from collections import deque
import math, random


def rt_curl(data_gen,acc):
    '''
    Apply curl field to accelerations for models that get state information

    ----------
    Parameters
    ----------
    data_gen - RandomTargetTimeseries class instance
    acc - torch.tensor
        has shape (batch_size,2). Accelerations for given timepoints
    '''
    if data_gen.curl:
        # if cursor location in curl field
        mask = ((acc[:,-2]>=data_gen.curl_xlims[0]) & (acc[:,-2]<=data_gen.curl_xlims[1])) | ((acc[:,-1]>=data_gen.curl_ylims[0]) & (acc[:,-1]<=data_gen.curl_ylims[1]))

        # apply field
        acc[mask,:] += torch.from_numpy(np.array([data_gen.curl_xmag, data_gen.curl_ymag])[np.newaxis,:])
    return acc

class data_buffer(object):
    '''
    General purpose buffer of fixed capacity with useful functionality
    '''
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self,obj):
        ''' append "obj" to buffer. If full drop oldest index'''
        self.buffer.append(obj)

    def sample(self, batch_size):
        '''sample uniformly "batch_size" samples from items in buffer, return as list'''
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        '''return current length of buffer'''
        return len(self.buffer)


class RandomTargetTimeseries():
    '''
    Generate random "center-out" reach timeseries for training and testing.
    '''
    def __init__(self,curl=False):
        # parameters for generating reaches
        self.max_pos = 50
        self.min_pos = -50
        self.max_target_acc = 1
        self.Tx = 100
        self.T_on = 20
        self.T_off = 100
        self.max_go_delay = 10
        self.width_go_cue=5
        self.tau=3 # time constant for generating sigmoidal movements
        self.scale_derivs = 10.

        # parameters for generating force perturbations
        self.curl=curl # bool for whether or not to apply curl field
        self.curl_xlims = [-25,25] # x limits of curl field
        self.curl_xmag=-5 # x-magnitude of curl field (spatial unit/time unit**2)
        self.curl_ymag = 0 # y-magnitude of curl field (spatial unit/time unit**2)
        self.curl_ylims = [0,50] # y limits of curl field


    def get_minibatch(self, m):
        """
        Generate minibatch of data
        ----------
        Parameters
        ----------
        m : int, minibatch size
        Returns
        -------
        X : torch.tensor
            has shape (m,3,Tx), 3 timeseries for target position and go signal
        Y : torch.tensor
            has shape (m,6,Tx), target kinematics timeseries
        CURL : torch.tensor
            has shape (m,6,Tx), peturbations to kinematics for curl field
        """


        # generate m random target positions
        targ_pos = (self.max_pos-self.min_pos)*np.random.rand(m,2)+self.min_pos
        # go-signal times
        go_time = self.T_on + np.random.randint(self.max_go_delay,size=[m,1])


        # create input timeseries
        X = np.zeros((m,3,self.Tx))
        X[:,:2,self.T_on:self.T_off] = targ_pos[:,:,np.newaxis]
        for i in range(self.width_go_cue): # Start cue
            X[:,2,go_time.ravel()+i] = 10

        # create output timeseries
        t = np.linspace(0,self.Tx,num=self.Tx)[np.newaxis,np.newaxis,:]
        pos = targ_pos[:,:,np.newaxis]/(1+np.exp(-(t-5*self.tau - go_time[:,:,np.newaxis])/self.tau)) # smooth sigmoidal reach
        vel = self.scale_derivs*np.diff(pos,prepend=0) # calculate velocity as derivative, scale up for training
        acc = self.scale_derivs*np.diff(vel,prepend=0) # calculate acceleration as second derivative
        Y = np.zeros((m,6,self.Tx))
        Y[:,:2,:]=acc
        Y[:,2:4,:]=vel
        Y[:,4:,:]=pos
        # Y = np.diff(np.diff(pos,prepend=0),prepend=0)


#         curl forces
        CURL = np.zeros((m,6,self.Tx))
        if self.curl:
            # apply x curl force
            curlx = CURL[:,0,:]
            curlx[(pos[:,0,:]>=self.curl_xlims[0]) & (pos[:,0,:]<=self.curl_xlims[1])] = self.curl_xmag

            # apply y curl force
            curly= CURL[:,1,:]
            curly[(pos[:,1,:]>=self.curl_ylims[0]) & (pos[:,1,:]<=self.curl_ylims[1])] = self.curl_xmag

            # apply velocity perturbations
            CURL[:,2,:]=np.cumsum(curlx,axis=-1)/self.scale_derivs
            CURL[:,3,:]=np.cumsum(curly,axis=-1)/self.scale_derivs

            # apply position perturbations
            CURL[:,4,:]=np.cumsum(np.cumsum(curlx,axis=-1),axis=-1)/self.scale_derivs/self.scale_derivs
            CURL[:,5,:]=np.cumsum(np.cumsum(curly,axis=-1),axis=-1)/self.scale_derivs/self.scale_derivs

        # create curl
        return torch.from_numpy(X).to(dtype=torch.float), torch.from_numpy(Y).to(dtype=torch.float), torch.from_numpy(CURL).to(dtype=torch.float)
