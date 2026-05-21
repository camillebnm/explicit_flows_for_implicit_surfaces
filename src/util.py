# coding: utf-8

import json
import math
import os
import os.path as osp
import shutil
import numpy as np
import torch
from warnings import warn
from src.diff_operators import gradient, mean_curvature
from src.meshing import create_mesh, save_ply, create_slice


def create_output_paths(checkpoint_path, experiment_name, overwrite=True,name = ""):
    """Helper function to create the output folders. Returns the resulting
    path.
    """
    full_path = os.path.join(".", checkpoint_path, experiment_name)
    full_path+="_" + name
    if os.path.exists(full_path) and overwrite:
        shutil.rmtree(full_path)
    elif os.path.exists(full_path):
        warn("Output path exists. Not overwritting.")
        return full_path

    os.makedirs(os.path.join(full_path, "models"))
    os.makedirs(os.path.join(full_path, "reconstructions"))
    os.makedirs(os.path.join(full_path, "kaolin"))
    os.makedirs(os.path.join(full_path, "summaries"))

    return full_path


def load_experiment_parameters(parameters_path):
    try:
        with open(parameters_path, "r") as fin:
            parameter_dict = json.load(fin)
    except FileNotFoundError:
        warn("File '{parameters_path}' not found.")
        return {}
    return parameter_dict


def reconstruct_at_times(model, times, meshpath, resolution=256, device="cpu", voxel_origin=[-1, -1, -1], voxel_dim=[2]*3,nom=None, resize=[1,1]):
    """Runs marching cubes on `model` at times `times`.

    Parameters
    ----------
    model: torch.nn.Module
        The model to run the inference. Must accept $\mathbb{R}^4$ inputs.

    times: collection of numbers
        The timesteps to use as input for `model`. The number of meshes
        generated will be `len(times)`.

    meshpath: str, PathLike
        Base folder to save all meshes.

    resolution: int, optional
        Marching cubes resolution. The input volume will have
        `resolution` ** 3 voxels. Default value is 256.

    device: str or torch.Device, optional
        The device where we will run the inference on `model`.
        Default value is "cpu".

    See Also
    --------
    nise.meshing.create_mesh
    """

    
    with torch.no_grad():
        k=0
        for t in times:
            if t is not None : 
               filename=osp.join(meshpath, f"{nom}_time_{k}.ply")
               k+=1
            else: filename = osp.join(meshpath, f"{nom}.ply")
            create_mesh(
                model,
                filename=filename,
                t=t,  # time instant for 4d SIREN function
                N=resolution,
                device=device,
                voxel_origin=voxel_origin,
                voxel_dim=voxel_dim, 
                resize=resize
            )
            
def reconstruct_sdf_at_times(model, times, meshpath, resolution=256, device="cpu", voxel_origin=[-1, -1, -1], voxel_dim=2,nom=None, resize=[1,1]):    
    with torch.no_grad():
        k=0
        for t in times:
            if t is not None : 
               filename=osp.join(meshpath, f"{nom}_time_{k}.pdf")
               k+=1
            else: filename = osp.join(meshpath, f"{nom}.pdf")
            create_slice(
                model,
                filename=filename,
                t=t,  # time instant for 4d SIREN function
                N=resolution,
                device=device,
                voxel_origin=voxel_origin,
                voxel_dim=voxel_dim, 
                resize=resize
            )
            

def estimate_differential_properties(
        model: torch.nn.Module, coords: torch.Tensor, with_curvs: bool = True,
        device: str = "cpu", batchsize: int = 10000
    ) -> np.ndarray:
    """Estimates gradient and curvature (optional) at `coords` using `model`.

    Parameters
    ----------
    model: torch.nn.Module
        The model to run the inference. Must accept $\mathbb{R}^4$ inputs.

    coords: numpy.ndarray
        The space-time coordinates to estimate the gradient and curvature on.

    with_curvs: bool, optional
        Whether to estimate the curvatures (True, default) or not (False).

    device: str or torch.Device, optional
        The device where we will run the inference on `model`.
        Default value is "cpu".

    batchsize: int, optional
        Number of points to perform the inference on at each step. We will
        iterate sequentially on the rows of `coords` running the inference on
        `batchsize` points. Default value is 10000. Tweak this to fit your
        specs.

    Returns
    -------
    verts: np.ndarray
        The output coords appended with normals and, optionally, mean
        curvature.

    See Also
    --------
    nise.diff_operators.gradient, nise.diff_operators.mean_curvature
    """
    model = model.eval()

    grads = torch.zeros_like(coords, device=device, requires_grad=False)
    if with_curvs:
        curvs = torch.zeros((coords.shape[0], 1), device=device, requires_grad=False)
    
    #computing the gradient in batches
    steps = int(math.ceil(coords.shape[0] / batchsize))
    for s in range(steps):
        a = s * batchsize
        b = (s+1) * batchsize
        out = model(coords[a:b, ...].unsqueeze(0).float())
        X = out['model_in']
        y = out['model_out']
        g = gradient(y, X)
        grads[a:b, ...] = g.detach().squeeze(0)
        if with_curvs:
            curvs[a:b, ...] = mean_curvature(g, X).detach().squeeze(0)

    verts = np.hstack((
        coords[..., :3].detach().cpu().numpy(),
        grads[..., :3].detach().cpu().numpy()
    ))
    if with_curvs:
        verts = np.hstack((
            verts,
            curvs.detach().cpu().numpy()
        ))

    return verts


def reconstruct_with_curvatures(model, times, meshpath, resolution=256,
                                device="cpu", batch_size=10000):
    attrs = [("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("quality", "f4")]
    model = model.eval()
    for t in times:
        verts, faces, _, _ = create_mesh(
            model,
            t=t,  # time instant for 4d SIREN function
            N=resolution,
            device=device
        )

        verts = torch.from_numpy(verts)
        coords = torch.cat((verts, t*torch.ones_like(verts[..., :1])), dim=1).squeeze(0).to(device)
        nsteps = int(math.ceil(verts.shape[0] / batch_size))
        grads = torch.zeros_like(coords)
        curvs = torch.zeros((grads.shape[0], 1))
        for s in range(nsteps):
            a = s * batch_size
            b = (s+1) * batch_size
            out = model(coords[a:b, ...].unsqueeze(0).float())
            X = out['model_in']
            y = out['model_out']
            g = gradient(y, X)
            c = mean_curvature(g, X)
            grads[a:b, ...] = g.detach().squeeze(0)
            curvs[a:b, ...] = c.detach().squeeze(0)

        verts = np.hstack((
            verts.detach().cpu().numpy(),
            grads[..., :3].detach().cpu().numpy(),
            curvs.detach().cpu().numpy()
        ))
        save_ply(
            verts=verts, faces=faces,
            filename=osp.join(meshpath, f"time_{t}.ply"),
            vertex_attributes=attrs
        )
def rigid_match(pc_source, pc_cible):
    Vz = torch.pca_lowrank(pc_cible)[2]
    Vy = torch.pca_lowrank(pc_source)[2]
    if Vy.det()<0: 
        Vy = -Vy
    if Vz.det()<0 : 
        Vz = -Vz
    T =  pc_cible.mean(axis=0) - pc_source.mean(axis=0)
    R = Vz @ Vy.T
    theta = torch.acos((R.trace()-1)/2)
    ind_min = torch.argmin(abs(torch.linalg.eig(R)[0] -1))
    u = torch.linalg.eig(R)[1][:,ind_min:(ind_min+1)].to(torch.float32)
    P = u@u.T
    Q = torch.tensor([[0, -u[2], u[1]],[u[2], 0, -u[0]],[-u[1],u[0] , 0]],device=pc_source.device)
    return theta, T, Q, P, u

def mat_from_theta(theta, Q, P):
    device=theta.device
    return (P.unsqueeze(-1) + torch.cos(theta)*(torch.eye(3,device=device)-P).unsqueeze(-1) +torch.sin(theta)*Q.unsqueeze(-1)).permute(2,0,1)#dxdxt

def eval_mass_sdf(inr, size = 1,n=100, device="cuda"):
  xs = torch.linspace(-size,size,n, device=device)
  ys = torch.linspace(-size,size,n, device = device)
  zs = torch.linspace(-size,size,n, device=device)
  X,Y = torch.meshgrid(xs,ys)
  coef = 0
  for z in zs : 
      data = torch.cat((X.reshape(n**2,1), Y.reshape(n**2,1),z*torch.ones(n**2,1,device=device)),axis=1).requires_grad_(True)
      coef += ((inr(data)["model_out"][:,0]<0)*1.).sum()
  return 8*size**3*coef/n**3

def scale(sdf, alpha): 
    for param in sdf.parameters(): 
        if param.shape[-1] ==3 : 
            param.data = param/alpha
        if param.shape[0] ==1 :
            param.data = param * alpha
            
def align_volume(sdf, V, eps,device): 
    m = eval_mass_sdf(sdf,size= 1, n=250, device=device)
    alpha = (V/m)**(1/3)
    print(f"init alpha : {alpha}")
    k=1
    while abs(m-V)>eps:
        scale(sdf, alpha)
        m = eval_mass_sdf(sdf,size= 1, n=250,device=device)
        scale(sdf,1/alpha)
        dm = m-V
        alpha = alpha - 1/(np.log(k+1)+0.1*torch.rand(1,device=device)**2)*dm
        print(dm)
        if alpha <0.2 : alpha = 0.99
        k+=1
    scale(sdf,alpha)
    return alpha ##La SDF est modifié sans instanciation explicit. ##
    
def scale_volume(sdf1, sdf2, eps, device):
    V1 = eval_mass_sdf(sdf1,size= 1, n=250,device=device)
    V2 = eval_mass_sdf(sdf2,size= 1, n=250, device=device)
    
    if V1>V2 : 
        first = True
        alpha = align_volume(sdf1, V2, eps, device=device)
    else : 
        alpha = align_volume(sdf2, V1, eps, device=device)
        first = False
    return alpha, first
    
def create_adjency_list(faces): 
    k = 0
    hl={}
    adjency = []
    for tri in faces : 
        tri = tri.sort()[0]
        for ind in [(0,1),(0,2),(1,2)]: 
            ind = (tri[ind[0]].item(), tri[ind[1]].item())
            if not ind in hl :
                hl[ind] = k
            else : 
                adjency.append([k, hl[ind]]) #Chaque element est une paire de triangle, rpz par leur indice. 

        k+=1
    #print(faces + list(set(second_list) - set(first_list)))
    return torch.tensor(adjency), hl
    
   
def compute_normales_faces(vert, faces): 
    normales_faces = []
    for tri in faces : 
        a , b, c =vert[tri]
        n = np.cross(a-b, a-c)
        normales_faces.append(n)
    normales_faces = np.asarray(normales_faces)
    nn = np.linalg.norm(normales_faces, axis=1, keepdims=True)
    mask = nn.squeeze()<10**-6
    normales_faces[mask]=0
    normales_faces[~mask]=normales_faces[~mask]/nn[~mask]
    return torch.tensor(normales_faces)
@torch.no_grad()
def mesh_sampler(n,vert, faces, normales_faces, f_area):
        rng = np.random.default_rng()
        W = rng.choice(faces.shape[0], n, p=f_area/f_area.sum(-1, keepdims=True))
        WW = np.unique(W, return_counts=True)
        plus_pc = np.ones((1,3))
        plus_n = np.ones((1,3))
        for i in range(WW[0].shape[0]) : 
            tri = faces[WW[0][i]]
            k = WW[1][i]
            if k >0 : 
                w = np.random.rand(3,k)
                w = w/w.sum(axis=0,keepdims=True)
                coord = vert[tri, :3]
                plus_pc = np.concatenate((plus_pc,(coord.T@w).T))
                plus_n = np.concatenate((plus_n, np.repeat(normales_faces[WW[0][i]:(WW[0][i]+1)],[k], axis=0))) 
        plus_pc = (plus_pc[1:]).reshape(n,3)
        plus_n = (plus_n[1:]).reshape(n,3)

        return plus_pc, plus_n

def read_corres(dataset, corres_path, size_sampled): 
    lms = []
    for i in range(len(dataset)): 
        c= []
        with open(corres_path[i], 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        c.append(int(line)-1)
                    
        c = (np.asarray(c)) 
        lms.append(dataset[i][c[:size_sampled], :4])
                    
    return lms
    
    
def anime_read( filename,device):
    """
    filename: path of .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: riangle face data of the 1st frame
        offset_data: 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    offset_data = np.concatenate((np.zeros((1,nv, 3)), offset_data))
    return nf, nv, nt, torch.tensor(vert_data,device=device).to(torch.float32), 
    
def print_shape_info(dataset, device): 
    pc_source = dataset.vertices_ni[0][0][: , :3]
    pc_cible = dataset.vertices_ni[1][0][ :, :3]
    sdf0 = dataset.vertices_ni[0][1]
    sdf1 = dataset.vertices_ni[1][1]
    m1 = eval_mass_sdf(sdf0, size = 1, n=250, device=device) 
    m2 = eval_mass_sdf(sdf1, size = 1, n=250, device=device) 
    print(f" masse 1: {m1}, masse 2 : {m2}")
    a1 = abs(dataset.vertices_ni[0][1](dataset.vertices_ni[0][0][ :1000, :3])["model_out"]).mean()
    a2 = abs(dataset.vertices_ni[1][1](dataset.vertices_ni[1][0][ :1000, :3])["model_out"]).mean()
    print(f"Diff masse : {m1-m2}")
    print(f"approx shape 1 : {a1}, approx shape 2 : {a2}", flush=True)
