import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import utilities as u


class BaselineRNN(nn.Module):
    '''
    Model to produce kinematics to reach to a randomly placed target.

    '''

    def __init__(self, inp_size=3, n_neurons=128, out_size=6, rnn_nonlinearity = 'relu',dropout_p=.3):
        '''
        Initialize layers to reuse during each timestep

        ----------
        Parameters
        ----------
        inp_size : int
                number of input timeseries, defaults to 3 (target x, target y, go cue)
        n_neurons : int
                number of neurons in each hidden layers

        '''
        super(BaselineRNN, self).__init__()



        # input -> ReLU RNN -> Dropout ->Linear -> ReLU -> Linear -> Kinematics (accx, accy, velx,vely,posx,posy)
        self.inp_size = inp_size
        self.out_size = out_size
        self.n_neurons = n_neurons

        # Recurrent layer
        self.rnn = nn.RNNCell(input_size=inp_size,hidden_size=n_neurons,nonlinearity=rnn_nonlinearity)

        # Output layer
        self.out_layer = nn.Sequential(nn.Dropout(p=dropout_p),nn.Linear(n_neurons, n_neurons), nn.ReLU(),nn.Linear(n_neurons, out_size))




    def forward(self, inp,  h_old):
        """
        Parameters
        ----------
        inp : torch.tensor, shape (batch_size,self.inp_size)
            Target position and go cue
        h_old : torch.tensor, shape (batch_size, self.n_neurons)
            previous hidden state of recurrent layer

        Returns
        -------
        kin : torch.tensor
            has shape (self.out_size,) corresponding to x and y acceleration.
        h_new : torch.tensor
            has shape (n_neurons,). new hidden state
        """


        # New hidden state
        h_new=self.rnn(inp,h_old)
        # Collect RNN output
        kin = self.out_layer(h_new)

        return kin, h_new


class RNN_Recursive(nn.Module):
    '''
    Network setup for recurrent control of the cursor
    '''
    def __init__(self, inp_size=3, n_neurons=128, out_size=2, rnn_type = 'relu',dropout_p=.3):
        '''
        Parameters
        ---------
        inp : int
            number of inputs, defaults to 3 (target locations and go cue)
        n_neurons: int
            number of units in hidden layers
        out_size: int
            number of variables to return
        rnn_type: string ['relu','tanh','gru','lstm']

        '''
        super(RNN_Recursive, self).__init__()


        self.inp_size = inp_size
        self.out_size = out_size
        self.n_neurons = n_neurons
        self.rnn_type = rnn_type


        # Input layers
        # Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout
        self.in_layer = nn.Sequential(nn.Linear(inp_size,n_neurons),nn.ReLU(),nn.Dropout(p=dropout_p),
                                    nn.Linear(n_neurons,n_neurons),nn.ReLU(),nn.Dropout(p=dropout_p))

        # Recurrent layer
        if rnn_type == "gru":
            self.rnn = nn.GRUCell(input_size=n_neurons,hidden_size=n_neurons)
        elif rnn_type=="lstm":
            raise NotImplementedError
        elif rnn_type in ['relu','tanh']:
            self.rnn = nn.RNNCell(input_size=n_neurons,hidden_size=n_neurons,nonlinearity=rnn_nonlinearity)
        else:
            raise NotImplementedError

        # Output layers
        # Dropout -> Linear -> ReLU -> Lienear -> Acc
        self.out_layer = nn.Sequential(nn.Dropout(p=dropout_p),nn.Linear(n_neurons, n_neurons), nn.ReLU(),nn.Linear(n_neurons, out_size))


    def forward(self, inp, h_old):
        """
        Parameters
        ----------
        inp : torch.tensor
            Hand and target positions. Has shape (7,).
            (go, hand_x, hand_y, curr_tgx, curr_tgy, next_tgx, next_tgy)
        h_old : torch.tensor
            Initial firing rates. Has shape (n_neurons,)
        task_info : torch.tensor
            tensor holding (go, curr_tgx, curr_tgy, next_tgx, next_tgy)

        Returns
        -------
        acc : torch.tensor
            has shape (2,) corresponding to x and y acceleration.
        h_new : torch.tensor
            has shape (n_neurons,), new hidden state of recurrent network
        """

        x = self.in_layer(inp)

        # Update RNN one time step.
        h_new = self.rnn(x, h_old)

        # Collect RNN output (acceleration of hand).
        acc = self.out_layer(h_new)

        return acc, h_new
