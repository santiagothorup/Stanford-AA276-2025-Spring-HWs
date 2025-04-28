import torch

"""For your convenience, we have provided f(x) from Homework 1."""
def f(x):
    """
    Return the control-independent part of the control-affine dynamics for the 13D quadrotor system.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
        
    returns:
        f: torch float32 tensor with shape [batch_size, 13]
    """
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

"""For your convenience, we have provided g(x) from Homework 1."""
def g(x):
    """
    Return the control-dependent part of the control-affine dynamics for the 13D quadrotor system.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
    returns:
        g: torch float32 tensor with shape [batch_size, 13, 4]
    """
    PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
    PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

    g = torch.zeros((*x.shape, 4))
    g[:, VXi, 0] = 2 * (QW*QY + QX*QZ)
    g[:, VYi, 0] = 2 * (QY*QZ - QW*QX)
    g[:, VZi, 0] = (1 - 2*torch.pow(QX, 2) - 2*torch.pow(QY, 2))
    g[:, WXi:, 1:] = torch.eye(3)

    return g

"""IMPLEMENT THE FOLLOWING FUNCTIONS"""

def optimal_control(x, dVdx):
    """
    Compute the optimal safety controller for the 13D quadrotor system for
    states x and
    value function spatial gradients dVdx.

    args:
        x:     torch tensor with shape [batch_size, 13]
        dVdx:  torch tensor with shape [batch_size, 13]
    
    returns:
        u_opt: torch tensor with shape [batch_size, 4]
    """
    # Optimal controller is a bang-ban controller for a control-affine system.
    batch_size = x.shape[0]
    
    u_opt = torch.zeros(batch_size, 4, dtype=torch.float32)
    
    u_upper = torch.tensor([20, 8, 8, 4], dtype=torch.float32)
    u_lower = torch.tensor([-20, -8, -8, -4], dtype=torch.float32)
    
    gx = g(x)
    
    for i in range(batch_size):
        g_i = gx[i]
        dVdx_i = dVdx[i]
        dVdx_g = torch.matmul(dVdx_i, g_i)
        
        for j in range(4):
            if dVdx_g[j] > 0:
                u_opt[i, j] = u_upper[j]
            elif dVdx_g[j] < 0:
                u_opt[i, j] = u_lower[j]
    return u_opt

def hamiltonian(x, dVdx):
    """
    Compute the Hamiltonian for the 13D quadrotor system for
    states x and
    value function spatial gradients dVdx.

    args:
        x:    torch tensor with shape [batch_size, 13]
        dVdx: torch tensor with shape [batch_size, 13]

    returns:
        ham:  torch tensor with shape [batch_size]
    """
    batch_size = x.shape[0]
    ham = torch.zeros(batch_size, dtype=torch.float32)
    u_upper = torch.tensor([20, 8, 8, 4], dtype=torch.float32)
    gx = g(x)
    fx = f(x)
    
    for i in range(batch_size):
        g_i = gx[i]
        f_i = fx[i]
        dVdx_i = dVdx[i]
        
        ham[i] = torch.matmul(dVdx_i, f_i) + torch.matmul(torch.abs(torch.matmul(dVdx_i, g_i)),u_upper)
    
    return ham

def hji_vi_loss(x, l, V, dVdt, dVdx):
    """
    Compute the HJI-VI loss term for the
    states x,
    failure function values l,
    value function values V,
    value function temporal gradients dVdt, and
    value function spatial gradients dVdx.
    This corresponds to h2(.) in Equation (14)
    in the DeepReach paper: https://arxiv.org/pdf/2011.02082.
    NOTE: You should return a batch of losses, i.e., you can
    interpret the ||.|| as |.| in Equation (14).

    args:
        x:    torch tensor with shape [batch_size, 13]
        l:    torch tensor with shape [batch_size]
        V:    torch tensor with shape [batch_size]
        dVdt: torch tensor with shape [batch_size]
        dVdx: torch tensor with shape [batch_size, 13] 

    returns:
        h2:   torch tensor with shape [batch_size]
    """
    batch_size = x.shape[0]
    h2 = torch.zeros(batch_size, dtype=torch.float32)
    ham = hamiltonian(x, dVdx)
    for i in range(batch_size):
        dVdt_i = dVdt[i]
        l_i = l[i]
        V_i = V[i]

        # Compute the HJI-VI 
        h2[i] = torch.abs(torch.min(l_i - V_i, dVdt_i + ham[i]))
    return h2