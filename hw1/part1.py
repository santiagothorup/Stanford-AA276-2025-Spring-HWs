"""
AA 276 Homework 1 | Coding Portion | Part 1 of 3


OVERVIEW

In this file, you will implement several functions required by the 
neural CBF library developed by the REALM Lab at MIT to
automatically learn your own CBFs for a 13D quadrotor system!

From this exercise, you will hopefully better understand the course
materials through a concrete example, appreciate the advantages
(and disadvantages) of learning a CBF versus manually constructing
one, and get some hands-on coding experience with using state-of-the-art
tools for synthesizing safety certificates, which you might find
useful for your own work.

If you are interested in learning more, you can find the library
here: https://github.com/MIT-REALM/neural_clbf


INSTRUCTIONS

Please refer to the Homework 1 handout for instructions and implementation details.

Function headers are provided below.
Your code should go into the sections marked by "YOUR CODE HERE"

When you are done, you can sanity check your code (locally) by running `python scripts/check1.py`.
After the tests pass, train a neural CBF (in your VM) by running `python scripts/train.py`.


IMPORTANT NOTES ON TRAINING
The training can take a substantial amount of time to complete [~9 hours ~= $10].
However, you should be able to implement all code for Parts 1, 2, and 3 in the meantime.
After each training epoch [50 total], the CBF model will save to 'outputs/cbf.ckpt'.
As long as you have at least one checkpoint saved [~10 minutes], Part 3 will load this checkpoint.
Try your best to not exceed $10 in credits -  you can stop training early if you reach this budget limit.

REMEMBER TO SHUTDOWN YOUR VIRTUAL MACHINES AFTER TRAINING, TO AVOID ACCUMULATING FEES.
"""

import torch
import numpy as np

BONUS = True 

def state_limits():
    """
    Return a tuple (upper, lower) describing the state bounds for the system.
    
    returns:
        (upper, lower)
            where upper: torch float32 tensor with shape [13]
                    lower: torch float32 tensor with shape [13]
    """
    if BONUS:
        upper = torch.tensor([6.28], dtype=torch.float32)
        lower = torch.tensor([-6.28], dtype=torch.float32)
    else:
        upper = torch.tensor([3,3,3,1,1,1,1,5,5,5,5,5,5], dtype=torch.float32)
        lower = torch.tensor([-3,-3,-3,-1,-1,-1,-1,-5,-5,-5,-5,-5,-5], dtype=torch.float32)
    
    return upper, lower


def control_limits():
    """
    Return a tuple (upper, lower) describing the control bounds for the system.
    
    returns:
        (upper, lower)
            where upper: torch float32 tensor with shape [4]
                lower: torch float32 tensor with shape [4]
    """
    if BONUS:
        upper = torch.tensor([3], dtype=torch.float32)
        lower = torch.tensor([-3], dtype=torch.float32)
    else:
        upper = torch.tensor([20, 8, 8, 4], dtype=torch.float32)
        lower = torch.tensor([-20, -8, -8, -4], dtype=torch.float32)
    
    return upper, lower


"""Note: the following functions operate on batched inputs.""" 


def safe_mask(x):
    """
    Return a boolean tensor indicating whether the states x are in the prescribed safe set.

    args:
        x: torch float32 tensor with shape [batch_size, 13]

    returns:
        is_safe: torch bool tensor with shape [batch_size]
    """
    if BONUS:
        # h(x) parameters
        a = 0.1  # rad
        b = np.sqrt(((3 - 20 * np.sin(a))/2) * a) # rad/s
        
        batch_size = x.shape[0]
        is_safe = torch.zeros(batch_size, dtype=torch.bool)
        for i in range(batch_size):
            safe = x[i,0]**2/(a**2) + x[i,1]**2/(b**2)
            if safe <= 1:
                is_safe[i] = True
            else:
                is_safe[i] = False
    else:
        # Safe set C = {x: sqrt(px^2 + py^2) > 2.8}
        Px = x[:, 0]
        Py = x[:, 1]
        
        batch_size = x.shape[0]
        is_safe = torch.zeros(batch_size, dtype=torch.bool)
        
        for i in range(batch_size):
            if (Px[i]**2 + Py[i]**2)**0.5 > 2.8:
                is_safe[i] = True
            else:
                is_safe[i] = False
    
    return is_safe


def failure_mask(x):
    """
    Return a boolean tensor indicating whether the states x are in the failure set.

    args:
        x: torch float32 tensor with shape [batch_size, 13]

    returns:
        is_failure: torch bool tensor with shape [batch_size]
    """
    if BONUS:
        batch_size = x.shape[0]
        is_failure = torch.zeros(batch_size, dtype=torch.bool)
        
        for i in range(batch_size):
            if np.abs(x[i:0]) > 0.3:
                is_failure[i] = True
            else:
                is_failure[i] = False
    else:
        # Failure set L = {x: sqrt(px^2 + py^2) < 0.5}
        Px = x[:, 0]
        Py = x[:, 1]
        
        batch_size = x.shape[0]
        is_failure = torch.zeros(batch_size, dtype=torch.bool)
        
        for i in range(batch_size):
            if (Px[i]**2 + Py[i]**2)**0.5 < 0.5:
                is_failure[i] = True
            else:
                is_failure[i] = False
    
    return is_failure


def f(x):
    """
    Return the control-independent part of the control-affine dynamics.
    Note: we have already implemented this for you!

    args:
        x: torch float32 tensor with shape [batch_size, 13]
        
    returns:
        f: torch float32 tensor with shape [batch_size, 13]
    """
    if BONUS:
        theta_i, theta_dot_i = [i for i in range(2)]
        theta, theta_dot = [x[:, i] for i in range(2)]
        
        g = 10
        l = 1
        
        f = torch.zeros_like(x)
        f[:, theta_i] = theta_dot
        f[:, theta_dot_i] = g/l * torch.sin(theta)
    else:
        PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
        PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

        f = torch.zeros_like(x)
        f[:, PXi] = VX
        f[:, PYi] = VY
        f[:, PZi] = VZ
        f[:, QWi] = -0.5*(WX*QX + WY*QY + WZ*QZ)
        f[:, QXi] =  0.5*(WX*QW + WZ*QY - WY*QZ)
        f[:, QYi] =  0.5*(WY*QW - WZ*QX + WX*QZ)
        f[:, QZi] =  0.5*(WZ*QW + WY*QX - WX*QY)
        f[:, VZi] = -9.8
        f[:, WXi] = -5 * WY * WZ / 9.0
        f[:, WYi] =  5 * WX * WZ / 9.0
    return f


def g(x):
    """
    Return the control-dependent part of the control-affine dynamics.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
    returns:
        g: torch float32 tensor with shape [batch_size, 13, 4]
    """
    if BONUS:
        theta_i, theta_dot_i = [i for i in range(2)]
        theta, theta_dot = [x[:, i] for i in range(2)]
                
        g_val = torch.zeros_like(x)
        g_val[:, theta_i] = 0
        g_val[:, theta_dot_i] = 0.5
    else:
        PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
        PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

        batch_size = x.shape[0]
        g = torch.zeros(batch_size, 13, 4, dtype=torch.float32)

        g[:, VXi, 0] = 2*(QW*QY + QX*QZ)
        g[:, VYi, 0] = 2*(QY*QZ - QW*QX)
        g[:, VZi, 0] = 2*(0.5 - QX**2 - QY**2)
        g[:, WXi, 1] = 1
        g[:, WYi, 2] = 1
        g[:, WZi, 3] = 1
    
    return g
