# coding: utf-8

import torch
from src.diff_operators import * 
import torch.autograd.forward_ad as fwAD

from src.nfgp_utils import *


class LossEdition(torch.nn.Module): 
    def __init__(self, ni, _, model,  char_list_loss, char_dict_weights):
        super().__init__()
        # Define the models
        self.sdf = ni[0]
        self.flot = model
        self.char_list_loss = char_list_loss
        self.char_dict_weights = char_dict_weights
        
    def recon(self,x, t=None): 
        if t is None : y = self.flot(x, -1) #Points starts at time 1 : pc_bac ...
        else : 
            y = self.flot(torch.cat((x[:,:-1],t),-1), -t)
        return self.sdf(y["model_out"][:,:-1])["model_out"], y
    
    def forward(self,  x, dummy1 , dummy2 , dummy3, dummy4, dummy5, dummy6, timespts, lms):
        return self.loss_edition( x, lms,  timespts)  

    def loss_edition(self, x, lms, timepts) : 
    
        loss_tot = {}
        log_det = {}
        
        if "D" in self.char_list_loss : 
            loss_lm = 0
            for i in range(2) : 
                advected_lm = self.flot(lms[i], 1-2*i)
                log_det["L"+str(i)] = advected_lm["log_det"]
                loss_lm += ((lms[1-i] - advected_lm["model_out"])**2).mean()
                
            loss_tot["loss_lm"] = loss_lm*self.char_dict_weights["D"]
            
            
        t=None
        if "T" in self.char_list_loss: 
            t = torch.rand(x.shape[0],1, device=x.device)
        
        if "B"  in self.char_list_loss or "S"  in self.char_list_loss : 
        
            y_out, x_inp = self.recon(x, t)
            J, J_status = jacobian_nfgp(x_inp["model_out"],x)
            yn, yn_proj = tangent_proj_mat(y_out, x_inp["model_out"])
            xn, xn_proj = tangent_proj_mat(y_out, x)
            
            J = torch.bmm(
                J[...,:-1 , :-1],
                xn_proj[...,:-1 , :-1]
            )

            J = addr(J,
                   yn[..., :-1],
                   xn[..., :-1])
                   
            weight = torch.abs(torch.linalg.det(J.view(x.shape[0], 3, 3)))
            weight = weight ** 2
            weight = 1. / weight.view(x.shape[0], 1, 1)
            #weight = weight / weight.sum(dim=-1, keepdim=True) * x.shape[0]


        if "B" in self.char_list_loss :

            y_out, x_inp = self.recon(x,t)
            
            h_out, J_status = hessian_nfgp(y_out, x)
            _, P = tangent_proj_mat(y_out, x)
            J, J_status = jacobian_nfgp(x_inp["model_out"], x)

            
            h_inp, J_status = hessian_nfgp(y_out, x_inp["model_out"])
            

            h_inp_J = torch.bmm(J[...,:-1 , :-1].transpose(1, 2).contiguous(), torch.bmm(h_inp[...,:-1 , :-1], J[...,:-1 , :-1]))
            diff = torch.bmm(
            P[...,:-1 , :-1].transpose(1, 2).contiguous(), torch.bmm(h_out[...,:-1 , :-1] - h_inp_J, P[...,:-1 , :-1]))
            
            loss_bend = ((diff*weight)**2).mean()
            loss_tot["Loss_bend"] = loss_bend*self.char_dict_weights["B"]

            
        
        if "S" in self.char_list_loss :
        
            y_out, x_inp = self.recon(x,t)
            _, P = tangent_proj_mat(y_out, x)
            J, J_status = jacobian_nfgp(x_inp["model_out"], x)

            
            I = torch.eye(3).view(1,3,3).to(J)
            diff = I - torch.bmm(J[...,:-1 , :-1].transpose(1, 2), J[...,:-1 , :-1])
            diff = torch.bmm(P[...,:-1 , :-1].transpose(1, 2), torch.bmm(diff, P[...,:-1 , :-1]))
            
            loss_stretch = ((diff*weight)**2).mean()
            loss_tot["Loss_stretch"] = loss_stretch*self.char_dict_weights["S"]
                
                      
        if "T" in self.char_list_loss : 
            with torch.no_grad(): 
                primal = timepts[:,-1:]
            tangent = torch.ones_like(primal) #torch.randn(t.shape,device=x.device)
            
            with fwAD.dual_level():
                dual_input = fwAD.make_dual(primal, tangent)
                dual_output,log_det_T = self.flot.forward_b(timepts[:,:-1], dual_input)
                log_det["T"] = log_det_T
                time_grad = fwAD.unpack_dual(dual_output).tangent
            loss_time = ((time_grad**2).sum(-1)).mean()
            loss_tot["Time_grad_loss"] = loss_time*self.char_dict_weights["T"]

                       
        return loss_tot
        
    def init_id(self, device): 
        optim = torch.optim.Adam(
            lr=1e-4,
            params=list(self.flot.parameters()),
            weight_decay=1e-1,
            )
        for i in range(500): 
            optim.zero_grad(set_to_none=True)
            pc = 6*torch.rand(1000, 3, device=device)-3 
            rand_time = 4*torch.rand(1000,1,device=device)-2
            pred, J = self.flot.forward_b(pc, rand_time)
            loss = ((pred-pc)**2).mean() + (J**2).mean()
            loss.backward()
            optim.step()
        print(f"Finale identity loss value : {loss}")

