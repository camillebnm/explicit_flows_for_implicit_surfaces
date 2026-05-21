# coding: utf-8

import torch
from src.diff_operators import * 
import torch.autograd.forward_ad as fwAD
from src.nfgp_utils import *


class LossMorphing(torch.nn.Module):
    """Morphing between two neural implict functions."""
    def __init__(self, model_list, time_list, flot, char_list_loss,char_dict_weights):
        super().__init__()
        # Define the models
        self.model1 = model_list[0]
        self.model2 = model_list[1]
        self.flot = flot
        self.t1 = time_list[0]
        self.t2 = time_list[1]
        self.cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.BCE = torch.nn.BCELoss(reduction='mean')
        self.char_list_loss = char_list_loss
        self.char_dict_weights = char_dict_weights
        print( char_dict_weights)
        print( char_list_loss ) 
        
    def recon_for(self,x, t=None): 
        if t is None : y = self.flot(x, -1) #Points starts at time 1 : pc_bac ...
        else : 
            y = self.flot(torch.cat((x[:,:-1],t),-1), -t)
        return self.model1(y["model_out"][:,:-1])["model_out"], y
        
    def recon_bac(self,x, t = None): 
        if t is None : y = self.flot(x, 1) #Points starts at time 0 : pc_for ...
        else : y = self.flot(torch.cat((x[:,:-1],1-t),-1), t)
        return self.model2(y["model_out"][:,:-1])["model_out"], y
    
    def init_id(self, device): 
        optim = torch.optim.Adam(
            lr=1e-4,
            params=list(self.flot.parameters()),
            weight_decay=1e-1,
            )
        for i in range(500): 
            optim.zero_grad(set_to_none=True)
            pc = torch.cat(((6*torch.rand(1000, 3, device=device)-3), torch.rand(1000, 1, device=device)), -1) 
            rand_time = 3*torch.rand(1000,1,device=device)-1
            pred= self.flot(pc, rand_time)
            if "log_det" in pred.keys():
                J = pred["log_det"]
                pred = pred["model_out"]
            else : 
                J = torch.zeros(1, 1, device=device)
                pred = pred["model_out"]
            loss = ((pred-pc)**2).mean() + (J**2).mean()
            loss.backward()
            optim.step()
        print(f"Finale identity loss value : {loss}")
        

    def loss_interpolation(self, pc_for, pc_bac, n_for, n_bac, sign_for, sign_bac, gt_sign, timepts, lms):
        
        loss_tot = {}
        
        mid_surf_time = torch.rand(pc_for.shape[0],1,device=pc_for.device)
        y_mid_for = self.flot(pc_for, mid_surf_time)
        y_mid_bac = self.flot(pc_bac, -mid_surf_time)
        
        mid_surf = torch.cat((y_mid_for["model_out"], y_mid_bac["model_out"]))
    
        if lms : 
            loss_tot["loss_lm"] =  self.char_dict_weights["D"]* (((model(lms[0],1)["model_out"]-lms[1])**2).mean() + ((model(lms[1],-1)["model_out"]-lm[0])**2).mean())
             #print(loss_lm)
        if "D" in self.char_list_loss :

            
            y_for = self.flot(y_mid_for["model_out"], 1-mid_surf_time)
            y_bac = self.flot(y_mid_bac["model_out"], mid_surf_time-1)
        
            #//Dicrichlet Loss//
            coords_for = y_mid_for["model_in"]
            pred_for = y_for["model_out"]
            val_for = self.model2(pred_for[...,:-1])["model_out"]

            coords_bac = y_mid_bac["model_in"]
            pred_bac = y_bac["model_out"]
            val_bac = self.model1(pred_bac[...,:-1])["model_out"]
            

            
            sdf_constraint = ((val_bac)**2).mean() + ((val_for)**2).mean()
            loss_tot["sdf_constraint"] = sdf_constraint*self.char_dict_weights["D"]

        



        if "N" in self.char_list_loss: 
            rec_for = self.recon_for(pc_bac) 
            rec_bac = self.recon_bac(pc_for)
            
            pred_n_bac =  gradient(rec_for[0], pc_bac)[...,:-1]
            pred_n_for =  gradient(rec_bac[0], pc_for)[...,:-1]
            
            
            signed_for = torch.sign( (pred_n_for*n_for).sum(-1, keepdim=True)) 
            signed_bac = torch.sign( (pred_n_bac*n_bac).sum(-1, keepdim=True))
            pred_n_bac = pred_n_bac * signed_bac
            pred_n_for = pred_n_for * signed_for
            
            neumann_constraint = ((1-self.cosim(pred_n_bac,n_bac))).mean() + ((1-self.cosim(pred_n_for,n_for))).mean()
            loss_tot["neumann_constraint"] = neumann_constraint*self.char_dict_weights["N"]
            

            
        #//Sign regu loss//
        if "S" in self.char_list_loss: 
            val_for, flot_for = self.recon_bac(sign_for)
            val_bac, flot_bac = self.recon_for(sign_bac)
            
            pred_sign = torch.cat((torch.sigmoid(5*val_for), torch.sigmoid(5*val_bac)))
            loss_sign = self.BCE(pred_sign, torch.sigmoid(5*gt_sign.unsqueeze(-1)) )  
            #mask = torch.cat((abs(flot_for).max(-1,keepdim=True)[0]<1.,abs(flot_bac).max(-1,keepdim=True)[0]<1. ))
            #loss_sign = (sign_diff/(0.01+abs(gt_sign.unsqueeze(-1)) )).mean()
            loss_tot["sign_constraint"] = loss_sign*self.char_dict_weights["S"]

                
        
        if "V" in self.char_list_loss and "T" in self.char_list_loss : 
            primal = timepts[:,-1:]
            tangent = torch.ones_like(timepts[:,-1:]) #torch.randn(t.shape,device=x.device)
            
            with fwAD.dual_level():
                dual_input = fwAD.make_dual(primal, tangent)
                dual_output,J_time = self.flot.forward_b(timepts[:,:-1], dual_input)
                time_grad = fwAD.unpack_dual(dual_output).tangent
 

            loss_time = ((time_grad**2).sum(-1)).mean()
            loss_tot["Time_grad_loss"] = loss_time*self.char_dict_weights["T"]
            loss_vp_time =  (J_time**2).mean()
            loss_tot["VP_in_time"] = loss_vp_time*self.char_dict_weights["V"]
            
        elif "T" in self.char_list_loss : 
            loss_time = 0
            for timepc in [timepts, mid_surf.detach().requires_grad_(True)] : 
                with torch.no_grad(): 
                    primal = 2*timepc[:,-1:]-1
                tangent = torch.ones_like(primal) #torch.randn(t.shape,device=x.device)
                
                with fwAD.dual_level():
                    dual_input = fwAD.make_dual(primal, tangent)
                    dual_output,_ = self.flot.forward_b(timepc[:,:-1], dual_input)
                    time_grad = fwAD.unpack_dual(dual_output).tangent
                loss_time += ((time_grad**2).sum(-1)).mean()
            loss_tot["Time_grad_loss"] = loss_time*self.char_dict_weights["T"]
            
        elif "V" in self.char_list_loss : 
            _, J_time = self.flot.forward_b(timepts[:,:-1],timepts[:,-1:])
            loss_vp_time =  (J_time**2).mean()
            loss_tot["VP_in_time"] = loss_vp_time*self.char_dict_weights["V"]
        
        if "I" in self.char_list_loss :
            loss_tot["Loss_jac"]=0
            for timepc in [timepts, mid_surf.detach().requires_grad_(True)] : 
                timepc = timepc.requires_grad_(True);
                j_out, _ = self.flot.forward_b(timepc[:,:-1],timepc[:,-1:])
                jac = jacobien(j_out, timepc)
                loss_iso = ((jac@jac.mT - torch.eye(3,device=timepc.device).view(1,3,3).repeat(timepc.shape[0], 1, 1))**2).mean()
                loss_tot["Loss_jac"] += loss_iso*self.char_dict_weights["I"]
            
        return loss_tot


    def forward(self, pc_for, pc_bac, n_for, n_bac, sign_for, sign_bac, gt_sign, timepts, lms ):
        return self.loss_interpolation(pc_for, pc_bac, n_for, n_bac, sign_for, sign_bac, gt_sign, timepts, lms)
        


