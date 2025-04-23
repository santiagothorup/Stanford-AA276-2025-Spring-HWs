"""
AA 276 Homework 1 | Coding Portion | Part 3 of 3


OVERVIEW

In this file, you will implement functions for 
visualizing your learned CBF from Part 1 and evaluating
the accuracy of the learned CBF and corresponding CBF-QP controller.


INSTRUCTIONS

Make sure you pass the tests for Part 1 and Part 2 before you begin.
Please refer to the Homework 1 handout for instructions and implementation details.

Function headers are provided below.
Your code should go into the sections marked by "YOUR CODE HERE"

When you are done, make sure that (in your VM) there is a CBF model checkpoint
saved at `outputs/cbf.ckpt`. Then, run `python scripts/plot.py`.
Submit the false safety rate reported in the terminal and the plot that is
saved to `outputs/plot.png`.

REMEMBER TO SHUTDOWN YOUR VIRTUAL MACHINES AFTER TRAINING, TO AVOID ACCUMULATING FEES.
"""


import torch
import numpy as np
import matplotlib as mpl

def plot_h(fig, ax, px, py, slice, h_fn):
    """
    Plot a 2D visualization of the CBF values across the grid defined by
    px and py for the provided state slice onto the provided matplotlib figure and axes.
    Note: px/py (x/y position state variable) defines the state grid along the
        x-axis/y-axis of the plot.
    Note: We will add plot titles/labels for you; you just need to add the
        colormap, its corresponding colorbar, and the zero level set contour.
        
    args:
        fig: matplotlib Figure
        ax: matplotlib Axes
        px: torch float32 tensor with shape [npx]
        py: torch float32 tensor with shape [npy]
        slice: torch float32 tensor with shape [13] (first 2 elements can be ignored)
        h_fn: Callable h=h_fn(x)
            h_fn takes a torch float32 tensor with shape [batch_size, 13]
            and outputs a torch float32 tensor with shape [batch_size]
    """
    # here is some starting code that might be helpful:
    PX, PY = torch.meshgrid(px, py)  
    X = torch.zeros((len(px), len(py), 13))
    X[..., 0] = PX
    X[..., 1] = PY
    X[..., 2:] = slice[2:]
    # X: torch float32 tensor with shape [len(px), len(py), 13] is a 2D grid of states
    # you should plot h_fn(X) (reshape X as needed to be compatible with h_fn)
    # you might want to use ax.pcolormesh(.), fig.colorbar(.), and ax.contour(.)

    npx, npy = PX.shape
    
    with torch.no_grad():
        h_vals = h_fn(X.reshape(-1, 13)).reshape(npx, npy)

    # Convert to NumPy 
    PX_np = PX.cpu().numpy()
    PY_np = PY.cpu().numpy()
    H_np  = h_vals.cpu().numpy()

    # Plot: heat-map, color-bar, and zero contour 
    cmap = mpl.cm.get_cmap("RdBu_r")     
    im   = ax.pcolormesh(PX_np, PY_np, H_np,
                        cmap=cmap, shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$h(x)$")

    # Zero level-set (CBF boundary)
    ax.contour(PX_np, PY_np, H_np,
            levels=[0.0], colors="k", linewidths=2)

    # Make the plot square in physical units
    ax.set_aspect("equal")



from part1 import safe_mask, failure_mask
from part2 import roll_out, u_qp


def plot_and_eval_xts(fig, ax, x0, u_ref_fn, h_fn, dhdx_fn, gamma, lmbda, nt, dt):
    """
    First, compute the state trajectories xts starting from initial states x0 under the CBF-QP
    controller given by the reference controller u_ref_fn, the CBF h_fn, the CBF gradient dhdx_fn, and
    parameters gamma and lmbda for nt Euler steps with time step dt.
    Hint: we have imported roll_out(.) and u_qp(.) from Part 2 for you to use.
        
    Next, plot the state trajectories xts projected onto the 2D position space on the provided matplotlib figure and axes.
        Note: We will add plot titles/labels for you; you just need to add the trajectories.

    Finally, return the false_safety_rate. Specifically, of the initial states x0 that are in the safe set,
        what proportion result in trajectories that actually violate safety?
    Hint: we have imported safe_mask(x) and failure_mask(x) from Part 1 for you to use.

    args:
        fig: matplotlib Figure
        ax: matplotlib Axes
        x0: torch float32 tensor with shape [batch_size, 13]
        u_ref_fn: Callable u_ref=u_ref_fn(x)
            u_ref_fn takes a torch float32 tensor with shape [batch_size, 13]
            and outputs a torch float32 tensor with shape [batch_size, 4]
        h_fn: Callable h=h_fn(x)
            h_fn takes a torch float32 tensor with shape [batch_size, 13]
            and outputs a torch float32 tensor with shape [batch_size]
        dhdx_fn: Callable dhdx=dhdx_fn(x)
            dhdx_fn takes a torch float32 tensor with shape [batch_size, 13]
            and outputs a torch float32 tensor with shape [batch_size, 13]
        gamma: float
        lmbda: float
        nt: int
        dt: float

    returns:
        false_safety_rate: float
    """
    # here is some starting code that defines the controller you should be using
    def u_fn(x):
        return u_qp(x, h_fn(x), dhdx_fn(x), u_ref_fn(x), gamma, lmbda)
    # first, you should compute state trajectories xts using roll_out(.)

    xts = roll_out(x0, u_fn, nt, dt)
    
    traj_np = xts.detach().cpu().numpy()     
    for traj in traj_np:
        ax.plot(
            traj[:, 0],                       # p_x
            traj[:, 1],                       # p_y
            lw=1.0,
            alpha=0.8,
            color="k",
        )
    
    batch_size = x0.shape[0]

    safe_init   = safe_mask(x0)        

    full_traj   = torch.cat([x0.unsqueeze(1), xts], dim=1) 
    fail_any    = failure_mask(full_traj.reshape(-1, 13))  
    fail_any    = fail_any.view(batch_size, nt + 1).any(dim=1) 

    false_safe  = safe_init & fail_any

    n_safe      = safe_init.sum().item()
    n_false     = false_safe.sum().item()
    false_safety_rate = float(n_false) / n_safe if n_safe > 0 else 0.0

    return false_safety_rate