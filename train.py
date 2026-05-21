#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import copy
import math
import os
import os.path as osp
import time
import sys
try:
    import kaolin
except ImportError:
    KAOLIN_AVAILABLE = False
else:
    KAOLIN_AVAILABLE = True
import numpy as np
import torch
import yaml
from src.dataset import SpaceTimePointCloudNI


from src.models.RealNVP import RealNVP

from src.util import create_output_paths, scale_volume, eval_mass_sdf, scale, read_corres, print_shape_info


def closure() : 
        # ===============================================================
        trainingpts= data["on_surf"][0] #:n_on_surface
        trainingnormals = data["on_surf"][1]
        trainingsdf = data["on_surf"][2]
        
        
        
        regusignpts = data["off_surf"][0]
        gt_sign = data["off_surf"][2].squeeze().detach().requires_grad_(True).clone()

        timepts= data["int_times"][0]
        
        adja_points = data["adja_faces"]

        gt = {
            "sdf": trainingsdf.float().unsqueeze(1),
            "normals": trainingnormals.float(),
        }
        optim.zero_grad(set_to_none=True)
        trainingpts = trainingpts + (2/1000)*torch.randn(trainingpts.shape[0], 4, device=device)/(e+1)

        pc_for = trainingpts[:int(trainingpts.shape[0]/2)].requires_grad_(True)
        pc_bac = trainingpts[int(trainingpts.shape[0]/2):].requires_grad_(True)
        
        sign_for = regusignpts[:int(regusignpts.shape[0]/2)].detach().requires_grad_(True)
        sign_bac = regusignpts[int(regusignpts.shape[0]/2):].detach().requires_grad_(True)

        n_for = trainingnormals[:int(trainingpts.shape[0]/2)]
        n_bac = trainingnormals[int(trainingpts.shape[0]/2):]

        loss = lossmorph(pc_for, pc_bac, n_for, n_bac,  sign_for, sign_bac, gt_sign, timepts, lms)
        l_loss.append(loss)
        running_loss = torch.zeros((1, 1), device=device).detach().requires_grad_(True)
        for k, v in loss.items():
            running_loss = running_loss + v
             

        running_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        running_losss.append(running_loss)
        return running_loss



if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "experiment_config", type=str, help="Path to the YAML experiment"
        " configuration file."
    )

    parser.add_argument(
        "--seed", default=668123, type=int,
        help="Seed for the random-number generator."
    )
    parser.add_argument(
        "--device", "-d", default="cuda:0", help="Device to run the training."
    )
    parser.add_argument(
        "--batchsize", "-b", default=0, type=int,
        help="Number of points to use per step of training. If set to 0,"
        " fetches it from the configuration file."
    )
    parser.add_argument(
        "--epochs", "-e", default=0, type=int,
        help="Number of epochs of training to perform. If set to 0, fetches it"
        " from the configuration file."
    )
    parser.add_argument(
        "--nom", default="",
        help="Add string at the end of the folder experience name"
    )
    parser.add_argument(
        "--load",type=str, default=None,
        help="Loading a pretrained model to improve convergence")
    parser.add_argument(
        "--landmark", default=None    )
    parser.add_argument(
        "--landmark_ind_based", default=None, type=float    )
    parser.add_argument(
        "--verbose", default=0, type=int    )
        
    def reload_quality(optim): 
        model.load_state_dict(best_weights)
        optim = torch.optim.Adam(
            lr=optim.param_groups[0]['lr']/10,
            params=list(model.parameters()),
            weight_decay=1e-5,
            )

        return optim
        
    def reload_explosion(optim): 
        model.load_state_dict(best_weights)
        optim = torch.optim.Adam(
            lr=optim.param_groups[0]['lr']/2,
            params=list(model.parameters()),
            weight_decay=1e-5,
            )

        return optim

    
    args = parser.parse_args()
    seed = torch.randint(1000000,(1,)).item()
    print( f"seed : {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    with open(args.experiment_config, 'r') as f:
        config = yaml.safe_load(f)
        
    devstr = args.device
    if "cuda" in args.device and not torch.cuda.is_available():
        print(f"[WARNING] Selected device {args.device}, but CUDA is not"
              " available. Using CPU", file=sys.stderr)
        devstr = "cpu"
    device = torch.device(devstr)

    training_config = config["training"]
    training_data_config = config["training_data"]
    training_mesh_config = training_data_config["mesh"]
    
    loss_config = config["loss"]
    loss_weights = loss_config.get("value")
    loss_terms = ""
    for k,v in loss_weights.items(): 
        loss_weights[k] = float(v)
        loss_terms +=k
    match loss_config["type"]: 
        case "morphing_sirens": 
            from src.losses.loss_morphing import LossMorphing as LossTraining
        case "edition":
            from src.losses.loss_edition import LossEdition as LossTraining

    epochs = training_config.get("n_epochs", 100)
    if args.epochs:
        epochs = args.epochs

    batchsize = training_data_config.get("batchsize", 20000)
    if args.batchsize:
        batchsize = args.batchsize

    meshdata = []
    ictimes = []
    corres_path = []
    for meshpath, data in training_mesh_config.items():
        ictimes.append(data['t'])
        meshdata.append((meshpath, data["ni"], data['t'], data["omega_0"]))
        if "corres" in data : 
            corres_path.append(data["corres"])

    dataset = SpaceTimePointCloudNI(meshdata, batchsize, size_domain = training_data_config["size"], device=device)

    edition_mode = loss_config["type"]=="edition"

    nsteps = epochs
    WARMUP_STEPS = int(nsteps/10)


    print(f"Total # of training steps = {nsteps}")

    network_config = config["network"]
    
    if args.verbose >= 1: 
        print_shape_info(dataset, device)

    lms = None
    if args.landmark : 
        print("loading landmarks : " + args.landmark)
        lm_dict = torch.load(args.landmark)
        lm0 = lm_dict["0"].to(device).to(torch.float32)#dataset.init_mesh[0][ lm_dict["0"], :4] 
        lm1 = lm_dict["1"].to(device).to(torch.float32)#dataset.init_mesh[1][ lm_dict["1"], :4]
        #Swap up the commented lines if you want to use indices based landmarks instead of 
        lms = [lm0, lm1]
    if args.landmark_ind_based : 
        print("Using points indices as landmark")
        size = min(dataset.init_mesh[0].shape[0], dataset.init_mesh[1].shape[0])
        size_sampled = int(args.landmark_ind_based*size)
        print(f"size of landmarks : {int(100*size_sampled/size)} %")
        lms = read_corres(dataset.init_mesh, corres_path, size_sampled)#Based on the Faust_r dataset format of landmarks. 
        lm0 = lms[0]
        lm1 = lms[1]
        
    if edition_mode : 
        lms = np.load(corres_path[0], allow_pickle=True).item()
        lm0 = torch.tensor(lms["handles"]).to(torch.float32)
        lm1 = torch.tensor(lms["targets"]).to(torch.float32)
        lm0 = torch.cat((lm0, torch.zeros(lm1.shape[0], 1)),-1).to(device)
        lm1 = torch.cat((lm1, torch.ones(lm1.shape[0], 1)),-1).to(device)
        lms = [lm0.requires_grad_(True), lm1.requires_grad_(True)]
        

    #Scaling
    if not edition_mode : 
        print("Post scaling")
        sdf0 = dataset.vertices_ni[0][1]
        sdf1 = dataset.vertices_ni[1][1]
        alpha, first = scale_volume(sdf0, sdf1, 10**-3, device=device)
        print(f"alpha : {alpha}")
        with torch.no_grad(): 
            if first : 
                dataset.vertices_ni[0][0][ :, :3] = alpha*dataset.vertices_ni[0][0][ :, :3]

            else : 
                dataset.vertices_ni[1][0][ :, :3] = alpha*dataset.vertices_ni[1][0][ :, :3]

                
        if args.landmark is not None or args.landmark_ind_based is not None : 
            lm0[:,:3] = lm0[:, :3] * (1- (1- alpha)*first)
            lm1[:,:3] = lm1[:, :3] * (alpha + (1- alpha)*first)
    
    if args.verbose>=2 : 
        import polyscope as ps
        ps.init()
        if args.landmark is not None or args.landmark_ind_based is not None : 
            nodes = torch.cat((lm0[:,:3].cpu(), lm1[:,:3].cpu()))
            edges = torch.arange(lm0.shape[0]).resize(lm0.shape[0],1)
            edges = torch.cat((edges, edges+lm0.shape[0]),-1)
            ps_net = ps.register_curve_network("my network", nodes, edges)
        ps.register_point_cloud("a", dataset.vertices_ni[0][0][ :, :3].detach().cpu())
        if not edition_mode : ps.register_point_cloud("b", dataset.vertices_ni[1][0][ :, :3].detach().cpu())
        ps.show()
        ps.remove_all_structures()

    #verification :
    if args.verbose>=1 and not edition_mode :
        print("Post scaling transformation verification")
        print_shape_info(dataset, device)
        if args.landmark is not None or args.landmark_ind_based is not None : 
            b1 = abs(dataset.vertices_ni[0][1](lm0[:,:3])["model_out"]).mean()
            b2 = abs(dataset.vertices_ni[1][1](lm1[:,:3])["model_out"]).mean()
            print(f"approx LM shape 1 : {b1}, approx shape 2 : {b2}", flush=True)   
            
    if edition_mode or args.landmark_ind_based or args.landmark : del lm0, lm1
    
    if network_config["base"]== "RealNVP" : 
        print("Using RealNVP")
        L_val = network_config["L"] if "L" in network_config.keys() is not None else 0
        model = RealNVP(L = L_val, style = network_config["style"], archi= network_config["archi"]).to(device)
        if args.load is not None : 
            print("Loading a pretrained model to improve convergence")
            model.load_state_dict(torch.load("results/"+args.load+"/models/best.pth"))
    if args.verbose >=1 : 
        print(model)
        
        x = dataset.vertices_ni[0][0][ :10000:10, :4]
        print(f"Numerical error on the composition law : {(model(model(x, 0.5)['model_out'], 0.5)['model_out'] - model(x, 1)['model_out']).norm()}")
        print(f"Numerical error on the invertibility property {(model(model(x, 0.5)['model_out'], -0.5)['model_out'] - x).norm()}")

    name = "edition" if edition_mode else "interpolation"        
    experiment = osp.split(args.experiment_config)[-1].split('.')

    if len(experiment) > 2 : 
        experiment = experiment[0]+experiment[1]
    else : 
        experiment = experiment[0]
    experimentpath = create_output_paths(
        "results",
        experiment,
        overwrite=False,
        name = name + args.nom
    )



    model.zero_grad(set_to_none=True)
        

    print("###############################")
    if config["optimizer"]["type"] =="adam" : 
        print("Selected optimizer : Adam")
        optim = torch.optim.Adam(
        lr=float(config["optimizer"]["lr"]),
        params=list(model.parameters()),
        weight_decay=1e-5,
        )

    else : 
        print("No optimizer selected, aborting ...")
        sys.exit(0)

    trainingpts = torch.zeros((batchsize, 4), device=device)
    trainingsdf = torch.zeros((batchsize), device=device)

    n_on_surface = batchsize
    
    allni = [vertni[1] for vertni in dataset.vertices_ni]
    lossmorph = LossTraining(allni, ictimes, model, loss_terms, loss_weights)

    checkpoint_times = training_config.get("checkpoint_times", ictimes)

    updated_config = copy.deepcopy(config)
    updated_config["network"]["init_method"] = "None"
    updated_config["training"]["n_epochs"] = epochs
    updated_config["training_data"]["batchsize"] = batchsize
    updated_config["training_data"]["n_on_surface"] = n_on_surface

    with open(osp.join(experimentpath, "config.yaml"), 'w') as f:
        yaml.dump(updated_config, f)
    best_loss = torch.inf
    best_weights = None

    training_loss = {}   
    start_training_time = time.time() 
    it_conv = 0
    

    if network_config["init"]=="identity" : 
        print("performing init to identity")
        lossmorph.init_id(device)
    else : 
        print("No training based init performed")
    
    print("Selected loss : " +loss_terms)    
    dict_loss = {} 
    reload_half = True
    losss = torch.inf
    threshold = 1.025

    best_weights = None
    

    print("start training")
    overfitting_mode = False
    for e in range(nsteps):
        running_losss = []
        l_loss = []
        data = dataset[e]
        
        optim.step(closure)

        
        if running_losss[-1].isnan(): 
            print("BEWARE nan appeared")
            break
        


        if best_loss > threshold*running_losss[-1].item() and e>3:
            dict_loss[e] = l_loss[-1]
            best_weights = copy.deepcopy(model.state_dict())
            best_loss = running_losss[-1].item()
            it_conv = e
            losss = l_loss[-1]

                
        if (best_loss < 1. or e/nsteps>0.8) and not overfitting_mode : 
            overfitting_mode = True
            print("Suffiscient quality has been obtained. Switching to overfitting")
            optim = reload_quality(optim)
            threshold = 1
            torch.save( best_weights, osp.join(experimentpath, "models", f"save_at_switching_{e}_{int(100*e/nsteps)}.pth") )

            
        if "sdf_constraint" in l_loss[-1].keys() and "Time_grad_loss" in l_loss[-1].keys(): 
            if l_loss[-1]["sdf_constraint"] < 0.01 and l_loss[-1]["Time_grad_loss"] < 0.1 : 
                print("training has converged, stopping")
                break
            
            
        
        if (100*e/nsteps)%2 < 99/nsteps and e > 0:
            print(f"Step {int(100*e/nsteps)}% --- Loss {running_losss[-1].item()}", flush = True)
            
        
        if running_losss[-1]>20*best_loss and  e>10 : 
                if best_weights is not None : 
                    print("Reloading model")
                    optim = reload_explosion(optim)
                    print(optim.param_groups[0]['lr'])
                    
                else : break
        if optim.param_groups[0]['lr'] < 10**-9 : 
            print("failed to converge")
            break


                
    print(f"final loss was :{losss}")
    training_time = time.time() - start_training_time
    print(f"training took {training_time} s")

    torch.save(
        model.state_dict(), osp.join(experimentpath, "models", "weights.pth")
    )
    torch.save(
        best_weights, osp.join(experimentpath, "models", "best.pth")
    )
    
    nd = {}
    for key in dict_loss.keys(): 
        sub_d = dict_loss[key]
        nd[key] = {}
        for k in sub_d.keys(): 
            val = sub_d[k].detach().cpu().item()
            nd[key][k] = val
    nd["time"] = training_time
    nd["%"] = it_conv/nsteps
    with open(osp.join(experimentpath,"loss.json"), 'w') as f:
        json.dump(nd, f)
    print(f"True number of itération : {int(100*it_conv/nsteps)}%")

