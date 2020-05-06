import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def createCenterOutTargets(r, N_TARGETS):
    target = torch.empty(N_TARGETS, 2)
    for i in range(N_TARGETS):
        target[i,0] = r * np.cos(np.pi*2*i/N_TARGETS)
        target[i,1] = r * np.sin(np.pi*2*i/N_TARGETS)
    return target

## The task class: tracks hand location, monitors is hand reaches the target, and controls the task to switch between delay period and movement period.
class PinballTask(nn.Module):
    def __init__(self, hand, screen_size, min_delay, max_delay, move_targ_thres, hold_targ_thres, padding):
        super(PinballTask, self).__init__()
        self.screen_size = screen_size
        self.padding = padding
        self.move_targ_thres = move_targ_thres
        self.hold_targ_thres = hold_targ_thres

        self.delay_left = np.random.uniform(min_delay, max_delay)
        self.delay = self.delay_left
        self.min_delay = min_delay
        self.max_delay = max_delay

        self.go = torch.tensor([False])

        hx, hy = hand
        self.curr_tgx = hx
        self.curr_tgy = hy
        #         self.next_tgx = padding + torch.rand(1) * (screen_size - 2 * padding)
        #         self.next_tgy = padding + torch.rand(1) * (screen_size - 2 * padding)
        self.target_index = 0
        self.next_tgx = TARGET_SEQUENCE[0, 0]
        self.next_tgy = TARGET_SEQUENCE[0, 1]

    def forward(self, hand):
    """
    Parameters
    ----------
    hand : torch.tensor
    Hand positions and velocities
    """

    # Compute distances to targets
    hx, hy = hand
    dist_to_curr = torch.sqrt(
        (hx - self.curr_tgx) ** 2 + (hy - self.curr_tgy) ** 2)
    dist_to_next = torch.sqrt(
        (hx - self.next_tgx) ** 2 + (hy - self.next_tgy) ** 2)

    # Reach period
    if self.go:
        # assert False

        # Check for target acquired
        if dist_to_next < self.move_targ_thres:
            self.go = torch.tensor([False])
            #self.target_index += 1
            self.target_index = torch.randint(low = 0, high = len(TARGET_SEQUENCE), size = (1,1))
            self.curr_tgx = self.next_tgx
            self.curr_tgy = self.next_tgy
            self.next_tgx = TARGET_SEQUENCE[self.target_index, 0]
            self.next_tgy = TARGET_SEQUENCE[self.target_index, 1]
            #                 self.next_tgx = self.padding + torch.rand(1) * (self.screen_size - 2 * self.padding)
            #                 self.next_tgy = self.padding + torch.rand(1) * (self.screen_size - 2 * self.padding)
            self.delay_left = np.random.uniform(self.min_delay, self.max_delay)

        # Delay / Hold period
    else:
        # Check if hold is violated
        if dist_to_curr >= self.hold_targ_thres:
            self.delay_left = np.random.uniform(self.min_delay, self.max_delay)
        else:
            self.delay_left -=1

        # Check if delay is done.
        if self.delay_left <= 0:
            self.go = torch.tensor([True])

    return torch.tensor([
    self.go,
    hx,
    hy,
    self.curr_tgx,
    self.curr_tgy,
    self.next_tgx,
    self.next_tgy
    ])[None, :]

class CenterOutTask(nn.Module):

    def __init__(self, hand, screen_size, min_delay, max_delay, move_targ_thres, hold_targ_thres, padding):
        super(CenterOutTask, self).__init__()
        self.screen_size = screen_size
        self.padding = padding
        self.move_targ_thres = move_targ_thres
        self.hold_targ_thres = hold_targ_thres

        self.delay_left = np.random.uniform(min_delay, max_delay)
        self.delay = self.delay_left
        self.min_delay = min_delay
        self.max_delay = max_delay

        self.go = torch.tensor([False])

        hx, hy = hand
        self.curr_tgx = torch.tensor([SCREEN_SIZE / 2])
        self.curr_tgy = torch.tensor([SCREEN_SIZE / 2])
        self.target_index = 0
        self.next_tgx = TARGET_SEQUENCE_CO[0, 0]
        self.next_tgy = TARGET_SEQUENCE_CO[0, 1]

    def forward(self, hand):
    """
    Parameters
    ----------
    hand : torch.tensor
        Hand positions and velocities
    """

        # Compute distances to targets
        hx, hy = hand
        dist_to_curr = torch.sqrt(
            (hx - self.curr_tgx) ** 2 + (hy - self.curr_tgy) ** 2)
        dist_to_next = torch.sqrt(
            (hx - self.next_tgx) ** 2 + (hy - self.next_tgy) ** 2)

        # Reach period
        if self.go:
            # assert False

            # Check for target acquired
            if dist_to_next < self.move_targ_thres:
                self.go = torch.tensor([False])
                #self.target_index += 1
                self.target_index = torch.randint(low = 0, high = len(TARGET_SEQUENCE_CO), size = (1,1))
                hx = torch.tensor([SCREEN_SIZE / 2])
                hy = torch.tensor([SCREEN_SIZE / 2])
                self.next_tgx = TARGET_SEQUENCE_CO[self.target_index, 0]
                self.next_tgy = TARGET_SEQUENCE_CO[self.target_index, 1]
                self.delay_left = np.random.uniform(self.min_delay, self.max_delay)

        # Delay / Hold period
        else:
            # Check if hold is violated
            if dist_to_curr >= self.hold_targ_thres:
                self.delay_left = np.random.uniform(self.min_delay, self.max_delay)
            else:
                self.delay_left -=1

            # Check if delay is done.
            if self.delay_left <= 0:
                self.go = torch.tensor([True])

        return torch.tensor([
            self.go,
            hx,
            hy,
            self.curr_tgx,
            self.curr_tgy,
            self.next_tgx,
            self.next_tgy
        ])[None, :]


## Compute losses based on hand location and the smoothness (penalize abrupt hand acceleration) of movement
class PinballCriterion(nn.Module):

    def __init__(self, acc_penalty):
        super(PinballCriterion, self).__init__()
        self.acc_penalty = acc_penalty

    def get_active_target(self, inp_hist):
        """Compute the active target at each timestep."""
        go = inp_hist[:, 0]
        curr_targ = inp_hist[:, 3:5]
        next_targ = inp_hist[:, 5:7]
        return curr_targ * (1 - go[:, None]) + next_targ * go[:, None]

    def target_loss(self, inp_hist):
        """Computes loss between hand and active target."""
        active_targ = self.get_active_target(inp_hist)
        return torch.mean((inp_hist[:, 1:3] - active_targ) ** 2)

    def acc_loss(self, acc_hist):
        """Computes penalty on the acceleration."""
        return self.acc_penalty * torch.mean(acc_hist ** 2)

    def forward(self, inp_hist, acc_hist):
        """
        Parameters
        ----------
        inp_hist : torch.tensor
            has shape (n_timesteps, 4) corresponding to (hx, hy, vx, vy) at each timepoint.
        acc_hist : torch.tensor
            has shape (n_timesteps, 2) corresponding to accelations
        """
        target_loss = self.target_loss(inp_hist)
        acc_loss = self.acc_loss(acc_hist)
        return target_loss + acc_loss
