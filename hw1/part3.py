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

    H = h_fn(X.reshape(-1, 13))              
    H = H.view_as(PX).detach().cpu()    

    Hplt, PXplt, PYplt = H.t(), PX.t().cpu(), PY.t().cpu()

    mesh = ax.pcolormesh(PXplt, PYplt, Hplt,cmap="viridis", shading="auto")
    fig.colorbar(mesh, ax=ax, label="h(x)")
    ax.contour(PXplt, PYplt, Hplt,levels=[0.0], colors="red", linewidths=1.0)




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

    # Roll out the system with the CBF-QP controller
    xts         = roll_out(x0, u_fn, nt, dt)
    B           = x0.shape[0]

    # ------------------------------------------------------------------
    # 3) Evaluate safety
    # ------------------------------------------------------------------
    safe_init   = safe_mask(x0)                                     # [B]
    left_safe   = (~safe_mask(xts.reshape(-1, 13)))                 # [B·nt]
    left_safe   = left_safe.view(B, nt).any(dim=1)                  # [B]

    false_safety_rate = (
        (safe_init & left_safe).float().sum().item() /
        max(safe_init.sum().item(), 1)                              # avoid /0
    )

    # ------------------------------------------------------------------
    # 4) Plot trajectories
    # ------------------------------------------------------------------
    px_traj = xts[:, :, 0].cpu().numpy()
    py_traj = xts[:, :, 1].cpu().numpy()

    for i in range(B):
        ax.plot(px_traj[i], py_traj[i],
                color = "red"  if left_safe[i] else "blue",
                linewidth = 1.0, alpha = 0.8)

    return false_safety_rate
