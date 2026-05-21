# coding: utf-8


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from src.models.Siren import SIREN

class Stat_Affine_Coupling_Sir(nn.Module):
    def __init__(self, mask, hidden_dim, depth, L): #L = 0 : no pos_embedding
        super(Stat_Affine_Coupling_Sir, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.mask = nn.Parameter(mask, requires_grad = False)
        self.L = L
        ## layers used to compute scale in affine transformation
        self.scale = SIREN(self.input_dim+3*self.L, self.input_dim, hidden_layer_config=[self.hidden_dim]*depth, w0=60)


        ## layers used to compute translation in affine transformation 
        self.translation = SIREN(self.input_dim+3*self.L, self.input_dim, hidden_layer_config=[self.hidden_dim]*depth, w0=60)

    def _compute_scale(self,x): 
        return self.scale(x)["model_out"]
    def _compute_translation(self, x): 
        return self.translation(x)["model_out"]
        
    def pos_enc(self, x, L): 
        return torch.cat((x, torch.sin(torch.tensor([2**n for n in range(L)], device=x.device)*x.unsqueeze(-1)).reshape(x.shape[0],3*L)), -1)


    
    def forward(self, x):
        ## convert latent space variable to observed variable
        var = self.pos_enc(x*self.mask, self.L)
        s = self._compute_scale(var)
        t = self._compute_translation(var)
        
        y = self.mask*x + (1-self.mask)*(x*torch.exp(s) + t)        
        logdet = torch.sum((1 - self.mask)*s, -1)
        
        return y, logdet

    def inverse(self, y):
        ## convert observed varible to latent space variable
        var = self.pos_enc(y*self.mask, self.L)
        s = self._compute_scale(var)
        t = self._compute_translation(var)
                
        x = self.mask*y + (1-self.mask)*((y - t)*torch.exp(-s))
        logdet = torch.sum((1 - self.mask)*(-s), -1)
        
        return x, logdet
        


        
full = [[1., 0, 1],
        [0., 1, 0],
        [0., 1, 1],
        [1., 0, 0],         
        [1., 1, 0],
        [0., 0, 1]]

simple = [[0., 1, 0],
        [1., 0, 0],         
        [0., 0, 1]]

mixed = [[1., 0, 1],
         [0., 1, 1],
        [1., 1, 0]]

class StatRealNVP(nn.Module):

    def __init__(self, hidden_dim = 128, L = 0, style = None,archi = ["full", 2, 2]):
        '''
        initialized with a list of masks. each mask define an affine coupling layer
        '''
        super(StatRealNVP, self).__init__()
        self.L = L  
        self.hidden_dim = hidden_dim 
        
        match archi[0]: 
            case "full": masks = full
            case "simple": masks = simple
            case "mixed" : masks = mixed
        masks = masks*int(archi[1])
        
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m),requires_grad = False)
             for m in masks])
        if style=="siren" : 
            print("Siren subnetwork")
            self.affine_couplings = nn.ModuleList(
            [Stat_Affine_Coupling_Sir(self.masks[i], self.hidden_dim, int(archi[2]),  self.L)
             for i in range(len(self.masks))])

        
    def forward(self, x):
        ## convert latent space variables into observed variables
        
        y = x
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot = logdet_tot + logdet

        ## a normalization layer is added such that the observed variables is within
        ## the range of [-4, 4].
        #logdet = torch.sum(torch.log(torch.abs(4*(1-(torch.tanh(y))**2))), -1)         
        #y = 4*torch.tanh(y)
        #logdet_tot = logdet_tot + logdet
        
        return y, logdet_tot

    def inverse(self, y):
        ## convert observed variables into latent space variables 
         
        x = y        
        logdet_tot = 0

        # inverse the normalization layer
        #logdet = torch.sum(torch.log(torch.abs(1.0/4.0* 1/(1-(x/4)**2))), -1)
        #x  = 0.5*torch.log((1+x/4)/(1-x/4))
        #logdet_tot = logdet_tot + logdet

        ## inverse affine coupling layers
        for i in range(len(self.affine_couplings)-1, -1, -1):
            x, logdet = self.affine_couplings[i].inverse(x)
            logdet_tot = logdet_tot + logdet
            
        return x, logdet_tot

    def reset_weights(self): 
        for mm in self.affine_couplings : 
            mm.scale.reset_weights()
            mm.translation.reset_weights()
