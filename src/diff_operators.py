# coding: utf-8

import torch
from torch.autograd import grad
import torch.autograd.forward_ad as fwAD


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def time_gradient(a, x, t): 
    primal = t
    tangent = torch.ones_like(t) #torch.randn(t.shape,device=x.device)
    def fn(t):
        return a(x, t)[0] #Remove the gradient computation of t+to wrt t wich is always 1. 

    with fwAD.dual_level():
        dual_input = fwAD.make_dual(primal, tangent)
        dual_output = fn(dual_input)
        jvp = fwAD.unpack_dual(dual_output).tangent
    
    return jvp
    
def gradient(y, x, grad_outputs=None):
    """Gradient of `y` with respect to `x`
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y,
        [x],
        grad_outputs=grad_outputs,
        create_graph=True
    )[0]
    return grad

def jacobien(V_out,input):
	GX = gradient(V_out[:,0],input).view(input.shape[0],1,input.shape[1])
	GY = gradient(V_out[:,1],input).view(input.shape[0],1,input.shape[1])
	GZ = gradient(V_out[:,2],input).view(input.shape[0],1,input.shape[1])
	DV = torch.cat((GX,GY,GZ),axis=1)[:,:,:-1]
	return DV

def mat_div(jac):
	div = []
	for mat in jac:
		div.append(torch.trace(mat))
	return torch.tensor(div)

def rotationnel(J,device):
  P = torch.tensor([[0.,1,0],[0,0,1],[1,0,0]],device=device)
  JJ1 = torch.matmul(J,P)
  JJ2 = torch.matmul(torch.swapaxes(J,1,2),P)
  d = torch.diagonal(JJ1-JJ2,dim1=1,dim2=2)
  s = torch.matmul(d,P)
  return s

def vector_dot(u, v):
    return torch.sum(u * v, dim=-1, keepdim=True)


def mean_curvature(grad, x):
    grad = grad[..., 0:3]
    grad_norm = torch.norm(grad, dim=-1)
    unit_grad = grad/grad_norm.unsqueeze(-1)

    Km = divergence(unit_grad, x)
    return Km
def laplacien_jac(J,x):
    Lx = divergence(J[:,0,:],x)
    Ly = divergence(J[:,1,:],x)
    Lz = divergence(J[:,2,:],x)
    DV = torch.cat((Lx,Ly,Lz),axis=1)
    return DV
    
def laplacien_jac_summed(J,x):
    DV = 0
    for i in range(3): 
        DV += divergence(J[:,i,:],x)**2
    return DV
