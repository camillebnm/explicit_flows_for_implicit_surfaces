# coding: utf-8


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from src.models.Siren import SIREN

class Affine_Coupling_Sir(nn.Module):
    def __init__(self, mask, hidden_dim, depth, L, omega): #L = 0 : no pos_embedding
        super(Affine_Coupling_Sir, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.mask = nn.Parameter(mask, requires_grad = False)
        self.L = L
        ## layers used to compute scale in affine transformation
        self.scale = SIREN(self.input_dim+3*self.L, self.input_dim-1, hidden_layer_config=[self.hidden_dim]*depth, w0=omega)


        ## layers used to compute translation in affine transformation 
        self.translation = SIREN(self.input_dim+3*self.L, self.input_dim-1, hidden_layer_config=[self.hidden_dim]*depth, w0=omega)

    def _compute_scale(self,x): 
        return self.scale(x)["model_out"]
    def _compute_translation(self, x): 
        return self.translation(x)["model_out"]
        
    def pos_enc(self, x, L): 
        return torch.cat((x, torch.sin(torch.tensor([2**n for n in range(L)], device=x.device)*x.unsqueeze(-1)).reshape(x.shape[0],3*L)), -1)


    
    def forward(self, x,time):
        ## convert latent space variable to observed variable
        var = torch.cat((self.pos_enc(x*self.mask[:-1], self.L),time*torch.ones(x.shape[0],1,device=x.device)), -1)
        s = self._compute_scale(var)
        t = self._compute_translation(var)
        
        y = self.mask[:-1]*x + (1-self.mask[:-1])*(x*torch.exp(s) + t)        
        logdet = torch.sum((1 - self.mask[:-1])*s, -1)
        
        return y, logdet

    def inverse(self, y,time):
        ## convert observed varible to latent space variable
        var = torch.cat((self.pos_enc(y*self.mask[:-1], self.L),time*torch.ones(y.shape[0],1,device=y.device)), -1)
        s = self._compute_scale(var)
        t = self._compute_translation(var)
                
        x = self.mask[:-1]*y + (1-self.mask[:-1])*((y - t)*torch.exp(-s))
        logdet = torch.sum((1 - self.mask[:-1])*(-s), -1)
        
        return x, logdet
        

class Affine_Coupling_Time(nn.Module):
    def __init__(self, mask, hidden_dim, L): #L = 0 : no pos_embedding
        super(Affine_Coupling_Time, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.mask = nn.Parameter(mask, requires_grad = False)
        self.L = L
        ## layers used to compute scale in affine transformation
        self.scale_fc1 = nn.Linear(self.input_dim+4*self.L, self.hidden_dim)
        self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc3 = nn.Linear(self.hidden_dim, self.input_dim-1)
        self.scale = nn.Parameter(torch.Tensor(self.input_dim-1))
        init.normal_(self.scale)

        ## layers used to compute translation in affine transformation 
        self.translation_fc1 = nn.Linear(self.input_dim + 4*self.L, self.hidden_dim)
        self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc3 = nn.Linear(self.hidden_dim, self.input_dim-1)
        
    def pos_enc(self, x, L): 
        return torch.cat((x, torch.sin(torch.tensor([2**n for n in range(L)], device=x.device)*x.unsqueeze(-1)).reshape(x.shape[0],4*L)), -1)

    def _compute_scale(self, x):
        ## compute scaling factor using unchanged part of x with a neural network
        s = torch.relu(self.scale_fc1(x))
        s = torch.relu(self.scale_fc2(s))
        s = torch.relu(self.scale_fc3(s)) * self.scale        
        return s

    def _compute_translation(self, x):
        ## compute translation using unchanged part of x with a neural network        
        t = torch.relu(self.translation_fc1(x))
        t = torch.relu(self.translation_fc2(t))
        t = self.translation_fc3(t)        
        return t
    
    def forward(self, x,time):
        ## convert latent space variable to observed variable
        var = self.pos_enc( torch.cat((x*self.mask[:-1],time*torch.ones(x.shape[0],1,device=x.device)), -1), self.L)
        s = self._compute_scale(var)
        t = self._compute_translation(var)
        
        y = self.mask[:-1]*x + (1-self.mask[:-1])*(x*torch.exp(s) + t)        
        logdet = torch.sum((1 - self.mask[:-1])*s, -1)
        
        return y, logdet

    def inverse(self, y,time):
        ## convert observed varible to latent space variable
        var = self.pos_enc( torch.cat((y*self.mask[:-1],time*torch.ones(y.shape[0],1,device=y.device)), -1), self.L)
        s = self._compute_scale(var)
        t = self._compute_translation(var)
                
        x = self.mask[:-1]*y + (1-self.mask[:-1])*((y - t)*torch.exp(-s))
        logdet = torch.sum((1 - self.mask[:-1])*(-s), -1)
        
        return x, logdet


class Affine_Coupling(nn.Module):
    def __init__(self, mask, hidden_dim, L): #L = 0 : no pos_embedding
        super(Affine_Coupling, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.mask = nn.Parameter(mask, requires_grad = False)
        self.L = L
        ## layers used to compute scale in affine transformation
        self.scale_fc1 = nn.Linear(self.input_dim+3*self.L, self.hidden_dim)
        self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc4 = nn.Linear(self.hidden_dim, self.input_dim-1)
        self.scale = nn.Parameter(torch.Tensor(self.input_dim-1))
        init.normal_(self.scale)

        ## layers used to compute translation in affine transformation 
        self.translation_fc1 = nn.Linear(self.input_dim + 3*self.L, self.hidden_dim)
        self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc4 = nn.Linear(self.hidden_dim, self.input_dim-1)
        
        self.act = torch.relu #torch.tanh #torch.nn.ELU(alpha=1.0, inplace=False) #
        
    def pos_enc(self, x, L): 
        return torch.cat((x, torch.sin(torch.tensor([2**n for n in range(L)], device=x.device)*x.unsqueeze(-1)).reshape(x.shape[0],3*L)), -1)

    def _compute_scale(self, x):
        ## compute scaling factor using unchanged part of x with a neural network
        s = self.act(self.scale_fc1(x))
        s = self.act(self.scale_fc2(s))
        s = self.act(self.scale_fc3(s))
        s = self.act(self.scale_fc4(s))*self.scale      
        return s

    def _compute_translation(self, x):
        ## compute translation using unchanged part of x with a neural network        
        t = self.act(self.translation_fc1(x))
        t = self.act(self.translation_fc2(t))
        t = self.act(self.translation_fc3(t))
        t = self.translation_fc4(t)        
        return t
    
    def forward(self, x,time):
        ## convert latent space variable to observed variable
        var = torch.cat((self.pos_enc(x*self.mask[:-1], self.L),time*torch.ones(x.shape[0],1,device=x.device)), -1)
        s = self._compute_scale(var)
        t = self._compute_translation(var)
        
        y = self.mask[:-1]*x + (1-self.mask[:-1])*(x*torch.exp(s) + t)        
        logdet = torch.sum((1 - self.mask[:-1])*s, -1)
        
        return y, logdet

    def inverse(self, y,time):
        ## convert observed varible to latent space variable
        var = torch.cat((self.pos_enc(y*self.mask[:-1], self.L),time*torch.ones(y.shape[0],1,device=y.device)), -1)
        s = self._compute_scale(var)
        t = self._compute_translation(var)
                
        x = self.mask[:-1]*y + (1-self.mask[:-1])*((y - t)*torch.exp(-s))
        logdet = torch.sum((1 - self.mask[:-1])*(-s), -1)
        
        return x, logdet
        
full = [[1., 0, 1, 1],
        [0., 1, 0, 1],
        [0., 1, 1, 1],
        [1., 0, 0, 1],         
        [1., 1, 0, 1],
        [0., 0, 1, 1]]

simple = [[0., 1, 0, 1],
        [1., 0, 0, 1],         
        [0., 0, 1, 1]]

mixed = [[1., 0, 1, 1],
         [0., 1, 1, 1],
        [1., 1, 0, 1]]

time_sparse = [[1., 0, 0, 1],
                [0., 1, 0, 1],
                [0., 0, 1, 1],
                [1., 0, 1, 0],
                [0., 1, 1, 0],    
                [1., 1, 0, 0],
                [1., 0, 1, 0],
                [0., 1, 0, 0],
                [0., 1, 1, 0],
                [1., 0, 0, 0],         
                [1., 1, 0, 0],
                [0., 0, 1, 0]]

class RealNVP(nn.Module):

    def __init__(self, hidden_dim = 128, L = 0, style = None,archi = ["full", 2, 2], omega=60):
        '''
        initialized with a list of masks. each mask define an affine coupling layer
        '''
        super(RealNVP, self).__init__()
        self.L = L  
        self.hidden_dim = hidden_dim 
        
        match archi[0]: 
            case "full": masks = full
            case "simple": masks = simple
            case "mixed" : masks = mixed
            case "time_sparse" : masks = time_sparse
        masks = masks*int(archi[1])
        
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m),requires_grad = False)
             for m in masks])
        if style=="siren" : 
            print("Siren subnetwork")
            self.affine_couplings = nn.ModuleList(
            [Affine_Coupling_Sir(self.masks[i], self.hidden_dim, int(archi[2]),  self.L, omega)
             for i in range(len(self.masks))])
        
        elif style=="time_embedding" :
            print("time embedding sub network")
            self.affine_couplings = nn.ModuleList(
            [Affine_Coupling_Time(self.masks[i], self.hidden_dim, self.L)
             for i in range(len(self.masks))])
        
        else : 
            print("classic real nvp")
            self.affine_couplings = nn.ModuleList(
            [Affine_Coupling(self.masks[i], self.hidden_dim, self.L)
             for i in range(len(self.masks))])
        
    def forward_b(self, x, t):
        ## convert latent space variables into observed variables
        
        y = x
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y,t)
            logdet_tot = logdet_tot + logdet

        ## a normalization layer is added such that the observed variables is within
        ## the range of [-4, 4].
        #logdet = torch.sum(torch.log(torch.abs(4*(1-(torch.tanh(y))**2))), -1)         
        #y = 4*torch.tanh(y)
        #logdet_tot = logdet_tot + logdet
        
        return y, logdet_tot

    def inverse(self, y, t):
        ## convert observed variables into latent space variables 
         
        x = y        
        logdet_tot = 0

        # inverse the normalization layer
        #logdet = torch.sum(torch.log(torch.abs(1.0/4.0* 1/(1-(x/4)**2))), -1)
        #x  = 0.5*torch.log((1+x/4)/(1-x/4))
        #logdet_tot = logdet_tot + logdet

        ## inverse affine coupling layers
        for i in range(len(self.affine_couplings)-1, -1, -1):
            x, logdet = self.affine_couplings[i].inverse(x,t)
            logdet_tot = logdet_tot + logdet
            
        return x, logdet_tot
        
    def forward(self, x, t): 
        xx = x[:,:-1]
        t0 = x[:,-1:]
        xx,J1 = self.inverse(xx,t0)

        xx,J2 = self.forward_b(xx, t+t0)
        return {"model_out" : torch.cat((xx,t+t0),axis=-1), "model_in" : x, "log_det": J1 + J2}
        
    def reset_weights(self): 
        for mm in self.affine_couplings : 
            mm.scale.reset_weights()
            mm.translation.reset_weights()
class DetailFlow(nn.Module): 
    def __init__(self, path, hidden_dim = 128, L = 0, style = None,archi = ["full", 2, 2]):
        '''
        initialized with a list of masks. each mask define an affine coupling layer
        '''
        super(DetailFlow, self).__init__()
        self.flow = RealNVP(L = 0, style = "siren", archi= ["full", 2, 2])
        self.flow.load_state_dict(torch.load(path))
        self.detail0 = RealNVP(L = L, style = style, archi= archi)
        self.detail1 = RealNVP(L = L, style = style, archi= archi)
        
        self.L = L  
        self.hidden_dim = hidden_dim 
        
    def forward(self, x, t): #NON a ne jamais appeller
        return self.flow(x,t)
    
    def reconstruct(self, x, t, way): 
        if way=="for": 
            if t < 1 : 
                y = self.flow(x, t)["model_out"]
            else : 
                y = self.flow(x, 1)["model_out"]
                y = self.detail1(y , t-1)["model_out"]
        if way=="bac": 
            if t < 0.2 : 
                y = self.detail1(x, -t)["model_out"]
            else : 
                y = self.detail1(x, -0.2)["model_out"]
                y = self.flow(y, -t+.2)["model_out"]
        return y
        
    def parameters(self): 
        return list(self.detail0.parameters()) + list(self.detail1.parameters())
        
    def to(self, device): 
        self.flow = self.flow.to(device)
        self.detail0 = self.detail0.to(device)
        self.detail1 = self.detail1.to(device)
        return self
        
        
