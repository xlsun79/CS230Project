import numpy as np
import torch
import torch.nn as nn

import utilities as u
import supervised_models as super_models # ;)



class baseline_model():
    '''
    class for training and testing basic RNN model
    '''
    def __init__(self,inp_size=3,n_neurons=128,out_size=6,rnn_nonlinearity = "relu",
                dropout_p=.3,curl=False,lr = 3E-4,BATCH_SIZE=32):

        '''
        initialize model and Parameters

        ----------
        Parameters
        ----------
        inp_size : int
                input size, defaults to 3 (target position and go cue)
        n_neurons: int
                number of hidden units in each layer
        out_size: int
                number of outputs, defaults to 6 (target kinematics in 2D)
        rnn_nonlinearity: string ['relu' or 'tanh']
                gru and lstm didn't improve performance so removed from class for now
        dropout_p: float [0-1]
                dropout percentage for fully connected layers
        curl: bool
                whether to include curl field
                to edit curl field parameters see utilities.RandomTargetTimeseries docstring
        lr : float
                learning rate for Adam optimizer
        BATCH_SIZE : int
                number of training examples in each mini-batch

        '''

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.losses = []
        self.model = super_models.BaselineRNN(inp_size=inp_size,n_neurons=n_neurons,out_size=out_size,
                                                rnn_nonlinearity=rnn_nonlinearity,
                                                dropout_p=dropout_p).to(self.device)
        self.data_gen = u.RandomTargetTimeseries(curl=curl) # data generator
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.BATCH_SIZE = BATCH_SIZE



    def train(self,N_TRAIN_EPOCHS,verbose=True):
        '''
        run backprop on N_TRAIN_EPOCHS mini-batches

        '''

        for epoch in range(int(N_TRAIN_EPOCHS)):
            X,Y,CURL = self.data_gen.get_minibatch(self.BATCH_SIZE) # get minibatch
            self.train_epoch(X.to(self.device),Y.to(self.device),CURL.to(self.device)) # push to GPU if available and run backprop
            if verbose:
                if epoch%10 == 0:
                    print("epoch",epoch,"loss",self.losses[-1])


    def train_epoch(self,X,Y,CURL):
        '''
        train a single epoch
        ----------
        Parameters
        ----------

        X : torch.tensor, size [batch size, number of inputs, timepoints]
            inputs
        Y: torch.tensor, size [batch size, number of outputs, timepoints]
            target timeseries
        CURL: torch.tensor, size [batch size, number of outputs, timepoints]
            curl forces
        '''

        self.model.train() # ensure dropout layers are turned on
        YHAT = torch.zeros(*Y.shape).to(self.device) # init estimates
        h= torch.zeros(self.BATCH_SIZE,self.model.n_neurons,dtype=torch.float).to(self.device) # init hidden state
        self.optimizer.zero_grad()
        for t in range(self.data_gen.Tx): # for each timepoint
            kin, h = self.model.forward(X[:,:,t],h) # forward prop
            YHAT[:,:,t]=kin + CURL[:,:,t]

        loss = (Y - YHAT).pow(2).mean() + (Y-YHAT).abs().mean() # elastic net like loss to get steeper gradient near zero
        loss.backward() # compute gradients
        self.optimizer.step() # backprop
        self.losses.append(loss.item()) # save losses


    def test_batch(self,m,to_cpu_return=True):
        '''
        evaluate model on "m" samples
        to_cpu_return : bool
                return outputs to cpu, if not using gpu, does nothing
        '''
        X,Y,CURL = self.data_gen.get_minibatch(m)
        X,Y,CURL = X.to(self.device),Y.to(self.device),CURL.to(self.device)
        YHAT = torch.zeros(*Y.shape).to(self.device)
        h= torch.zeros(m,self.model.n_neurons,dtype=torch.float).to(self.device)

        self.model.eval()
        for t in range(self.data_gen.Tx):
            with torch.no_grad():
                kin, h = self.model.forward(X[:,:,t],h)
            YHAT[:,:,t]=kin + CURL[:,:,t]

        if to_cpu_return:
            return X.to("cpu"),Y.to("cpu"),YHAT.to("cpu"),CURL.to("cpu")
        else:
            return X,Y,YHAT,CURL


class recursive_model():
    '''
    class for training and testing recursive RNN model for cursor control.
    This network outputs x and y cursor accelerations.
    The input to this network is target location, go cue, and a buffer of recent
    outputs from the network.
    '''

    def __init__(self,inp_size=3,n_neurons=128,out_size=2,rnn_type = "gru",
                dropout_p=.3,curl=False,lr = 3E-4,BATCH_SIZE=32,BUFFER_CAP=5):
        '''
        initialize model and Parameters

        ----------
        Parameters
        ----------
        inp_size : int
                input size without buffer, defaults to 3 (target position and go cue)
        n_neurons: int
                number of hidden units in each layer
        out_size: int
                number of outputs, defaults to 6 (target kinematics in 2D)
        rnn_nonlinearity: string ['relu', 'tanh', 'gru']
                lstm not implemented yet
        dropout_p: float [0-1]
                dropout percentage for fully connected layers
        curl: bool
                whether to include curl field
                to edit curl field parameters see utilities.RandomTargetTimeseries docstring
        lr : float
                learning rate for Adam optimizer
        BATCH_SIZE : int
                number of training examples in each mini-batch
        BUFFER_CAP : int
                capacity of recent output buffer, determines input size to network
        '''

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.losses = []
        self.model = super_models.RNN_Recursive(inp_size=inp_size + 2*BUFFER_CAP,n_neurons=n_neurons,out_size=out_size,
                                                rnn_type=rnn_type,
                                                dropout_p=dropout_p).to(self.device)
        self.data_gen = u.RandomTargetTimeseries(curl=curl)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_CAP = BUFFER_CAP


    def train(self,N_TRAIN_EPOCHS,verbose=True,elastic_net_lamb=.5):
        '''
        run backprop on N_TRAIN_EPOCHS mini-batches
        elastic_net_lamb : float
                    mixing parameter for l1 and l2 loss

        '''

        for epoch in range(int(N_TRAIN_EPOCHS)):
            X,Y,_ = self.data_gen.get_minibatch(self.BATCH_SIZE) # generate data
            # train epoch (keep only acceleration infor for training)
            self.train_epoch(X.to(self.device),Y[:,:2,:].to(self.device),elastic_net_lamb)
            if verbose:
                if epoch%100 == 0:
                    print("epoch",epoch,"loss",self.losses[-1])


    def train_epoch(self,X,Y,elastic_net_lamb):
        '''
        train a single epoch
        ----------
        Parameters
        ----------

        X : torch.tensor, size [batch size, number of inputs, timepoints]
            inputs
        Y: torch.tensor, size [batch size, number of outputs, timepoints]
            target timeseries
        elastic_net_lamb : float
            mixing parameter for l1 and l2 loss
        '''

        self.model.train() # ensure dropout is turned on
        YHAT = torch.zeros(*Y.shape).to(self.device) # init estimates
        h= torch.zeros(self.BATCH_SIZE,self.model.n_neurons,dtype=torch.float).to(self.device) # init hidden state
        self.optimizer.zero_grad() # clear gradients

        kin_buff = u.data_buffer(self.BUFFER_CAP) # fill acceleartion buffer with zeros
        for b in range(self.BUFFER_CAP):
          kin_buff.push(torch.zeros(self.BATCH_SIZE,2,dtype=torch.float).to(self.device))

        for t in range(self.data_gen.Tx):
            INPUT = torch.cat([X[:,:,t],*kin_buff.buffer],dim=1) # concatenate targets and buffer
            kin,h = self.model.forward(INPUT,h)  # forward pass
            kin = u.rt_curl(self.data_gen,kin) # apply curl field
            kin_buff.push(kin) # push outputs to buffer
            YHAT[:,:,t]=kin # save outputs

        V,VHAT = torch.cumsum(Y,dim=2),torch.cumsum(YHAT,dim=2)/self.data_gen.Tx # estimated velocity and target velocity
        D,DHAT = torch.cumsum(V,dim=2),torch.cumsum(VHAT,dim=2)/self.data_gen.Tx # estimated position and target position

        # loss is on position from acceleration output
        loss = elastic_net_lamb*(D-DHAT).pow(2).mean()+(1-elastic_net_lamb)*(D-DHAT).abs().mean() #+

        loss.backward() # calculate gradients
        self.optimizer.step() # backprop
        self.losses.append(loss.item()) # save losses

    def test_batch(self,m,to_cpu_return=True):
        '''
        evaluate model on "m" samples
        to_cpu_return : bool
                return outputs to cpu, if not using gpu, does nothing
        '''

        self.model.eval() # turn off dropout
        X,Y,_ = self.data_gen.get_minibatch(m) # get test data
        X,Y = X.to(self.device),Y[:,:2,:].to(self.device)
        YHAT = torch.zeros(*Y.shape).to(self.device)

        h= torch.zeros(m,self.model.n_neurons,dtype=torch.float).to(self.device)
        kin_buff = u.data_buffer(self.BUFFER_CAP) # fill acceleration buffer
        for b in range(self.BUFFER_CAP):
          kin_buff.push(torch.zeros(m,2,dtype=torch.float).to(self.device))

        for t in range(self.data_gen.Tx):
            with torch.no_grad(): # don't track gradients
                INPUT = torch.cat([X[:,:,t],*kin_buff.buffer],dim=1) # concat targets and buffer
                kin,h = self.model.forward(INPUT,h) # forward prop
                kin = u.rt_curl(self.data_gen,kin)

            kin_buff.push(kin) # push outputs to buffer
            YHAT[:,:,t]=kin

        if to_cpu_return:
            return X.to("cpu"),Y.to("cpu"),YHAT.to("cpu")
        else:
            return X,Y,YHAT
