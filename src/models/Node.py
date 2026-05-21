# coding: utf-8


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from torchdiffeq import odeint_adjoint as odeint
from src.models.Siren import SIREN


class ODEFunc(nn.Module): 
    def __init__(self, net):
            super(ODEFunc, self).__init__()
            
    
            self.net = net
    
    def forward(self,  t, y):

        return torch.cat((self.net(y)["model_out"], torch.ones(y.shape[0], 1, device=y.device)), -1)   

class NODE(nn.Module):

    def __init__(self, device):
        super(NODE, self).__init__()
        
        siren = SIREN(4, 3, hidden_layer_config=[256]*6,
                         w0=30).to(device)
        odefunc = ODEFunc(siren)

        self.net = odefunc
        
    def forward_b(self, x, t): 
        return self.net(t, x), None

    def forward(self, y, t):
    
        t = t*torch.ones(y.shape[0], 1, device=y.device)
        t = torch.cat((torch.zeros(1, device=y.device), t.reshape(t.shape[0])[0:1], ))
        if t[-1] == 0 : 
            return {"model_out" : y, "model_in" : (y, t)}
        else : 
            return {"model_out" : odeint(self.net, y, t)[1], "model_in" : (y, t)}
        
class ODEAdj(nn.Module):

    def __init__(self,net):
        super(ODEAdj, self).__init__() 
        self.net = net

    def forward(self, y, t):
        t = torch.cat((torch.zeros(1, device=y.device), t.reshape(t.shape[0])[0:1], ))
        
        return {"model_out" : odeint_adjoint(self.net, y, t)[1], "model_in" : (y, t)}

