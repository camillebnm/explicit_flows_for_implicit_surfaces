#!/usr/bin/env python
# coding: utf-8



import argparse
import os
import os.path as osp
import torch
import re
import yaml
from src.models.Siren import from_pth
from src.util import reconstruct_at_times, scale_volume

from src.models.RealNVP import RealNVP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run marching cubes using a trained model."
    )
    parser.add_argument(
        "exp_path", nargs='+',
        help="Path to the experiment folder"
    )
    parser.add_argument(
        "--modes", nargs='+',
        help="render modes"
    )
    parser.add_argument(
        "--resolution", "-r", default=128, type=int,
        help="Resolution to use on marching cubes. Default is 128"
    )
    parser.add_argument(
        "--times", "-t", nargs='+', default=[-1, 0, 1],
        help="Parameter values to run inference on. Default is [-1, 0, 1]."
    )
    parser.add_argument(
        "--nom", default = "")
    
 
    args = parser.parse_args()
    
    devstr = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(devstr)
    L_config = []
    for i in range(len(args.exp_path)):
        with open(args.exp_path[i]+"config.yaml", 'r') as f:
            L_config.append(yaml.safe_load(f))
    is_pairwise = len(args.exp_path) > 1 
    is_edition = L_config[0]["loss"]["type"]=="edition"
    
    path = re.split("/", args.exp_path[0])

    exp_elements = re.split("_", path[-2]) #0: morph #1: name1-name2 / name #3: type exp

    if is_pairwise: 
        path_pairwise = re.split("/", args.exp_path[1])
        exp_elements_pairwise = re.split("_", path_pairwise[-2])
        shaperef, shape0 = re.split("-", exp_elements[1])
        shaperef_bis, shape1 = re.split("-", exp_elements_pairwise[1])
        if not shaperef == shaperef_bis : 
            print("No pairwise possible in this case")
            exit(0)
        out_dir = args.exp_path[0]+"outputs_pairwise_with_"+path_pairwise[-2]
        shape_list = [shaperef, shape0, shape1]

      
    elif is_edition: 
        shape0 = exp_elements[1]
        out_dir = args.exp_path[0]+"outputs"
        shape_list = [shape0]
    else : 
        shape0, shape1 = re.split("-", exp_elements[1]) 
        out_dir = args.exp_path[0]+"outputs"
        shape_list = [shape0, shape1]
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir)
    
    WWs = {}
    for i in range(len(args.exp_path)) :       
        for key in L_config[i]["training_data"]["mesh"].keys():
            for shape in shape_list : 
                WWs[shape] = L_config[i]["training_data"]["mesh"][key]["omega_0"]
    
    
    models_parameter = [torch.load(path+"models/best.pth") for path in args.exp_path]
    models_list = []
    for i in range(len(args.exp_path)) :  
        if L_config[i]["network"]["base"] =="RealNVP": 
            L = L_config[i]["L"] if "L" in L_config[i]["network"].keys() else 0
            model = RealNVP(L=L, style  = L_config[i]["network"]["style"]).to(device)
            model.load_state_dict(models_parameter[i])
            models_list.append(model)
    
    sdf_list = [from_pth("ni/"+sdf_path+".pth", w0=WWs[sdf_path]).to(device) for sdf_path in shape_list]
    
    scale_factor = [scale_volume(sdf_list[0], sdf_list[i], 10**-3, device) for i in range(len(args.exp_path))] if not is_edition else [(1,1)] 
    
    alpha = 1
    if len(scale_factor)>2: 
        if scale_factor[1][1] : alpha = 1/scale_factor[1][0]
    else : alpha = scale_factor[0][0]
    
    print(f"Running marching cubes running with resolution {args.resolution}")
    if args.times[0] =="None" :
        times = [None]
    elif args.times[0] =="linspace":
        times = [t.item() for t in torch.linspace(0.,1., int(args.times[1]))]
    else  : times = [float(t) for t in args.times]
    
    
    
    if is_pairwise: 
        def forw(x): 
            xx = x[:,:-1]
            t = x[:,-1:]
            p = torch.cat((xx, torch.ones(x.shape[0],1,device=x.device)),-1)
            advection  = models_list[0](alpha*models_list[1](p,-t)["model_out"], t)["model_out"]
            return sdf_list[1](advection[:,:-1])
    
        def backw(x):
            xx = x[:,:-1]
            t = x[:,-1:]
            p = torch.cat((xx, torch.ones(x.shape[0],1,device=x.device)),-1)
            advection  = models_list[0](alpha*models_list[1](p,-t)["model_out"], t-1)["model_out"]
            advection2 = models_list[1](1/alpha*advection, 1)["model_out"]
            return sdf_list[2](advection2[:,:-1])
            
        def meanw(x): 
            t = x[:,-1:]
            return {"model_out" : (1-t)*forw(x)["model_out"] + t*backw(x)["model_out"]} 
    else : 
        def forw(x): 
            xx = x[:,:-1]
            t = x[:,-1:]
            p = torch.cat((xx, t*torch.ones(x.shape[0],1,device=x.device)),-1)
            return sdf_list[0](models_list[0](p, -t)["model_out"][:,:-1])
        def backw(x):
            xx = x[:,:-1]
            t = x[:,-1:]
            p = torch.cat((xx, (1-t)*torch.ones(x.shape[0],1,device=x.device)),-1)
            return sdf_list[1](models_list[0](p, t)["model_out"][:,:-1])
            
        def meanw(x): 
            xx = x[:,:-1]
            t = x[:,-1:]
            p1 = torch.cat((xx, t*torch.ones(x.shape[0],1,device=x.device)),-1)
            p2 = torch.cat((xx, (t)*torch.ones(x.shape[0],1,device=x.device)),-1)
            return {"model_out" : (1-t)*sdf_list[0](model(p1, -t)["model_out"][:,:-1])["model_out"] + t*sdf_list[1](model(p2, 1-t)["model_out"][:,:-1])["model_out"]}


    if "forw" in args.modes : 
        reconstruct_at_times(forw, times, out_dir, resolution=args.resolution, device=device, nom="forw" + args.nom, voxel_origin = [-1.2]*3, voxel_dim=[2.4]*3)
    if "backw" in args.modes : 
        reconstruct_at_times(backw, times, out_dir, resolution=args.resolution, device=device, nom="backw" + args.nom , voxel_origin = [-1.2]*3, voxel_dim=[2.4]*3)
    if "meanw" in args.modes : 
        reconstruct_at_times(meanw, times, out_dir, resolution=args.resolution, device=device, nom="meanw" + args.nom , voxel_origin = [-1.2]*3, voxel_dim=[2.4]*3)
    print("Done")
