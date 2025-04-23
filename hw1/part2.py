"""
AA 276 Homework 1 | Coding Portion | Part 2 of 3


OVERVIEW

In this file, you will implement functions for simulating the
13D quadrotor system discretely and computing the CBF-QP controller.


INSTRUCTIONS

Make sure you pass the tests for Part 1 before you begin.
Please refer to the Homework 1 handout for instructions and implementation details.

Function headers are provided below.
Your code should go into the sections marked by "YOUR CODE HERE"

When you are done, you can sanity check your code (locally) by running `python scripts/check2.py`.
"""


import torch
from part1 import f, g


"""Note: the following functions operate on batched inputs."""


def euler_step(x, u, dt):
    """
    Return the next states xn obtained after a discrete Euler step
    for states x, controls u, and time step dt.
    Hint: we have imported f(x) and g(x) from Part 1 for you to use.
    
    args:
        x: torch float32 tensor with shape [batch_size, 13]
        u: torch float32 tensor with shape [batch_size, 4]
        dt: float
        
    returns:
        xn: torch float32 tensor with shape [batch_size, 13]
    """
    gxu = torch.bmm(g(x), u.unsqueeze(2)).squeeze(2)
    xn = x + (f(x) + gxu) * dt
    return xn


def roll_out(x0, u_fn, nt, dt):
    """
    Return the state trajectories xts obtained by rolling out the system
    with nt discrete Euler steps using a time step of dt starting at
    states x0 and applying the controller u_fn.
    Note: The returned state trajectories should start with x1; i.e., omit x0.
    Hint: You should use the previous function, euler_step(x, u, dt).

    args:
        x0: torch float32 tensor with shape [batch_size, 13]
        u_fn: Callable u=u_fn(x)
            u_fn takes a torch float32 tensor with shape [batch_size, 13]
            and outputs a torch float32 tensor with shape [batch_size, 4]
        nt: int
        dt: float

    returns:
        xts: torch float32 tensor with shape [batch_size, nt, 13]
    """
    batch_size = x0.shape[0]
    xts = torch.zeros((batch_size, nt, 13), dtype=torch.float32)
    x = x0.clone()
    for i in range(nt):
        u = u_fn(x)
        x = euler_step(x, u, dt)
        xts[:, i, :] = x
    return xts

import cvxpy as cp
import numpy as np
from part1 import control_limits


def u_qp(x, h, dhdx, u_ref, gamma, lmbda):
    """
    Return the solution of the CBF-QP with parameters gamma and lmbda
    for the states x, CBF values h, CBF gradients dhdx, and reference controls u_nom.
    Hint: consider using CVXPY to solve the optimization problem: https://www.cvxpy.org/version/1.2/index.html
        Note: We are using an older version of CVXPY (1.2.1) to use the neural CBF library.
            Make sure you are looking at the correct version of documentation.
        Note: You may want to use control_limits() from Part 1.
    Hint: If you use multiple libraries, make sure to properly handle data-type conversions.
        For example, to safely convert a torch tensor to a numpy array: x = x.detach().cpu().numpy()

    args:
        x: torch float32 tensor with shape [batch_size, 13]
        h: torch float32 tensor with shape [batch_size]
        dhdx: torch float32 tensor with shape [batch_size, 13]
        u_ref: torch float32 tensor with shape [batch_size, 4]
        gamma: float
        lmbda: float

    returns:
        u_qp: torch float32 tensor with shape [batch_size, 4]
    """
    # Convert inputs to numpy arrays
    x_np      = x.detach().cpu().numpy()      
    h_np      = h.detach().cpu().numpy()    
    dhdx_np   = dhdx.detach().cpu().numpy()    
    u_ref_np  = u_ref.detach().cpu().numpy()    
    
    # Convert control limits to numpy arrays
    u_upp, u_low = control_limits()  
    u_upp = u_upp.detach().cpu().numpy()   # upper bounds
    u_low = u_low.detach().cpu().numpy()     # lower bounds
    
    # Defining f(x) and g(x) 
    f_val = f(x).detach().cpu().numpy()
    g_val = g(x).detach().cpu().numpy()
    
    batch_size = x_np.shape[0]
    
    # Initialize u_qp
    u_qp_np = np.zeros((batch_size, 4))
    
    for i in range(batch_size):
        # Define variables
        u_i = cp.Variable(4)
        delta_i = cp.Variable()
        
        # Calculating lie derivatives
        Lf_h_i = np.dot(dhdx_np[i], f_val[i])
        Lg_h_i = dhdx_np[i].dot(g_val[i])
        
        # Calculating h_dot
        h_dot_i = Lf_h_i + cp.sum(cp.multiply(Lg_h_i, u_i))
        
        # Defining Constraints
        constraints = [
            u_low <= u_i, u_i <= u_upp, # Control limits
            h_dot_i + gamma*h_np[i] + delta_i >= 0,  # CBF condition
            delta_i >= 0  # Non-negativity of delta
        ]
        
        # Defining Objective 
        objective = cp.Minimize(cp.sum_squares(u_i - u_ref_np[i]) + lmbda * cp.square(delta_i))
        
        # Defining Problem 
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        u_qp_np[i,:] = u_i.value
    return torch.tensor(u_qp_np, dtype=torch.float32)