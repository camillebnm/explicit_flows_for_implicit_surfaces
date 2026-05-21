# coding: utf-8
import math
import numpy as np
from plyfile import PlyData
import torch
from torch.utils.data import Dataset
from src.util import compute_normales_faces
from src.models.Siren import SIREN, from_pth
from src.diff_operators import gradient
from scipy.spatial import  KDTree


def _sample_on_surface(vertices: torch.Tensor, n_points: int, device: str):
    """Samples row of a torch tensor containing vertices of a surface.

    Parameters
    ----------
    vertices: torch.Tensor
        A mode-2 tensor where each row is a vertex.

    n_points: int
        The number of points to sample. If `n_points` >= `vertices.shape[0]`,
        we simply return `vertices` without any change.

    device: str or torch.device
        The device where we should generate the indices of sampled points.
        Ideally, this is the same device where `vertices` is stored.

    Returns
    -------
    sampled: torch.tensor
        The points sampled from `vertices`. If
        `n_points` == `vertices.shape[0]`, then we simply return `vertices`.

    idx: torch.tensor
        The indices of points sampled from `vertices`. Naturally, these are
        row indices in `vertices`.

    See Also
    --------
    torch.randperm, torch.arange
    """
    if n_points >= vertices.shape[0]:
        return vertices, torch.arange(end=n_points, step=1, device=device)
    idx = torch.randperm(vertices.shape[0], device=device)[:n_points]
    sampled = vertices[idx, ...]
    return sampled, idx


def _sample_initial_condition(
    vertices: torch.tensor,   # pass a list of tensors (vertices), one for each mesh
    n_on_surf: int,
    n_off_surf: int,
    ni: torch.nn.Module,      # Same here
    device: torch.device = torch.device("cpu"),
    no_sdf: bool = False,
):
    """Creates a set of training data with coordinates, normals and SDF
    values.

    Parameters
    ----------
    vertices: torch.tensor
        A mode-2 tensor with the mesh vertices.

    n_on_surf: int
        # of points to sample from the mesh.

    n_off_surf: int
        # of points to sample from the domain. Note that we sample points
        uniformely at random from the domain.

    ni: torch.nn.Module
        Neural Implicit Open3D raycasting scene to use when querying SDF
        for domain points.

    device: str or torch.device, optional
        The compute device where `vertices` is stored. By default its
        torch.device("cpu")

    no_sdf: boolean, optional
        Don't query SDF for domain points, instead we mark them with SDF = -1.

    Returns
    -------
    coords: dict[str => list[torch.Tensor]]
        A dictionary with points sampled from the surface (key = "on_surf")
        and the domain (key = "off_surf"). Each dictionary element is a list
        of tensors with the vertex coordinates as the first element of said
        list, the normals as the second element, finally, the SDF is the last
        element.

    See Also
    --------
    sample_on_surface, curvature_segmentation
    """
    surf_pts, _ = _sample_on_surface(
        vertices,
        n_on_surf,
        device=device
    )

    coord_dict = {
        "on_surf": [surf_pts[..., :4],   # x, y, z, t
                    surf_pts[..., 4:7],  # nx, ny, nz
                    surf_pts[..., -1]]   # sdf
    }

    if n_off_surf != 0:
        domain_pts = torch.rand((n_off_surf, 3), device=device).requires_grad_(True) * 2 - 1
        # domain_pts = off_surf_sampler.sample((n_off_surf, 3)).to(device)
        t = surf_pts[0, 3]  # We assume that all points in surf_pts have the same value of t.
        if no_sdf is False:
            out = ni(domain_pts)
            domain_sdf = out["model_out"]
            domain_normals = gradient(domain_sdf, out["model_in"]).detach()
            domain_sdf = domain_sdf.detach()
        else:
            domain_sdf = torch.full(
                (n_off_surf, 1),
                fill_value=-1,
                device=device
            )
            domain_normals = -torch.ones_like(domain_pts, device=device)

        coord_dict["off_surf"] = [
            torch.column_stack((
                domain_pts,
                torch.full_like(domain_pts, fill_value=t, device=device)[..., 0]
            )),  # x, y, z, t
            domain_normals,
            domain_sdf.squeeze()
        ]

    return coord_dict


def _create_training_data(
    initial_conditions: list,
    n_samples: int,
    off_surface_sampler: torch.distributions.distribution.Distribution,
    device: torch.device,
    time_sampler: torch.distributions.distribution.Distribution = None,
    fraction_on_surface: float = 0.25,
    fraction_off_surface: float = 0.25
):
    """Samples a batch of training points.

    Parameters
    ----------
    initial_conditions: list[torch.Tensor, torch.nn.Module]
        A list with all initial conditions. Each initial condition contains
        a tensor with the mesh vertices and a neural implicit representation to
        estimate the SDF values.

    n_samples: int
        The total number of samples to draw.

    off_surface_sampler: torch.distributions.distribution.Distribution
        The distribution to draw samples for off-surface point coordinates at
        intermediate-times

    device: torch.device
        The device to store any tensors created.

    time_sampler: torch.distributions.distribution.Distribution
        The distribution to draw samples for off-surface parameter values at
        intermediate times. Default value is None, meaning that
        `off_surface_sample` will be used for this as well.

    fraction_on_surface: float
        Fraction of points to be drawn from the initial condition vertices.

    fraction_off_surface: float
        Fraction of points to be drawn off-surface from the initial conditions.
        intermediate time points as well.

    Returns
    -------
    full_samples: dict[str => list[torch.Tensor]]
        A dictionary with three keys: "on_surf", "off_surf", "int_times". Each
        element is an ordered list of tensors, where the first element is the
        point coordinates, followed by the normals and, the SDF values.
    """
    n_on_surface = math.ceil(n_samples * fraction_on_surface)
    n_off_surface = math.floor(n_samples * fraction_off_surface)
    n_int_times = n_samples - (n_on_surface + n_off_surface)

    if len(initial_conditions) > 1:
        n_on_surface = n_on_surface // len(initial_conditions)
        n_off_surface = n_off_surface // len(initial_conditions)

    full_samples = []

    for vertices, ni, _, _ , _ in initial_conditions:
        samples = _sample_initial_condition(
            vertices,
            n_on_surf=n_on_surface,
            n_off_surf=n_off_surface,
            ni=ni,
            device=device
        )
        if not full_samples:
            full_samples = samples
        else:
            for k in full_samples:
                for i in range(len(full_samples[k])):
                    full_samples[k][i] = torch.cat((
                        full_samples[k][i], samples[k][i]
                    ), dim=0)

    if n_int_times:
        int_pts = None
        if time_sampler is None:
            int_pts = off_surface_sampler.sample((n_int_times, 4)).to(device)
        else:
            int_pts = off_surface_sampler.sample((n_int_times, 3)).to(device)
            times = time_sampler.sample((n_int_times,)).to(device)
            int_pts = torch.column_stack((int_pts, times))

        full_samples["int_times"] = [
            int_pts,
            -torch.ones((n_int_times, 3), dtype=torch.float32, device=device),
            -torch.ones((n_int_times,), dtype=torch.float32, device=device)
        ]
    else:
        full_samples["int_times"] = []
        
    full_samples["adja_faces"] = []

    return full_samples


def _read_ply(path: str, t: float):
    """Reads a PLY file with position and normal data.

    Note that we expect the input ply to contain x,y,z vertex data, as well
    as nx,ny,nz normal data. The time coordinate is added as a column in the
    returned `vertices` tensor.

    Parameters
    ----------
    path: str, PathLike
        Path to the ply file. We except the file to be in binary format.

    t: number
        The parameter value for this mesh.

    Returns
    -------
    mesh: o3d.t.geometry.TriangleMesh
        The fully constructed Open3D Triangle Mesh. By default, the mesh is
        allocated on the CPU:0 device, since Open3D still doesn't support GPU
        nearest-neighbor operations.

    vertices: torch.Tensor
        The same vertex information as stored in `mesh`, augmented by the SDF
        values as the last column (a column of zeroes) and time data in column
        3. Returned for easier, structured access.

    See Also
    --------
    PlyData.read, o3d.t.geometry.TriangleMesh
    """
    # Reading the PLY file and adding the time info
    n_columns = 8  # x, y, z, t, nx, ny, nz, sdf
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=(num_verts, n_columns), dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        # column 3 is time
        if t != 0:
            vertices[:, 3] = t
        vertices[:, 4] = plydata["vertex"].data["nx"]
        vertices[:, 5] = plydata["vertex"].data["ny"]
        vertices[:, 6] = plydata["vertex"].data["nz"]

        faces = np.stack(plydata["face"].data["vertex_indices"])

    # Converting the PLY data to open3d format
    #device = o3c.Device("CPU:0")
    #mesh = o3d.geometry.TriangleMesh(device)
    #mesh.vertex["positions"] = o3c.Tensor(vertices[:, :3], dtype=o3c.float32)
    #mesh.vertex["normals"] = o3c.Tensor(vertices[:, 3:6], dtype=o3c.float32)
    #mesh.triangle["indices"] = o3c.Tensor(faces, dtype=o3c.int32)
    return torch.from_numpy(faces).requires_grad_(False).to(torch.int32), torch.from_numpy(vertices).requires_grad_(False)


class SpaceTimePointCloudNI(Dataset):
    """Space-time varying point clouds with Neural Implicits for SDF querying.

    Parameters
    ----------
    inputpaths: list[(str, str, number, number)]
        List of tuples with paths to the base meshes (PLY format only), their
        neural implicit (NI) representations, the parameter value
        (-1 <= t <= 1) for each mesh, and omega_0 value for the NI.

    batchsize: int
        # of points to sample at each call to `__getitem__`.

    device: torch.device
        Device to store the NIs and vertex data read. By default, we store
        them on "cuda:0"

    fraction_on_surface: number, optional
        Fraction of points to sample from the initial conditions' surface per
        each batch. By default we sample 1/4 of points from the meshes at each
        call to `__getitem__`

    fraction_off_surface: number, optional
        Fraction of points to sample from the initial conditions' domain, i.e.
        off-surface points per batch. By default we sample 1/4 of points from
        the meshes' domains at each call to `__getitem__`

    See Also
    --------
    SpaceTimePointCloud

    References
    ----------
    Tiago Novello, Guilherme Schardong, Luiz Schirmer, Vinícius da Silva,
    Hélio Lopes, and Luiz Velho. Exploring differential geometry in neural
    implicits. Computers & Graphics, 108, 2022
    """
    def __init__(self, inputpaths, batchsize, device=torch.device("cuda:0"),
                 fraction_on_surface=0.45, fraction_off_surface=0.45, size_domain = 1, something = 0., hd = False):
        super(SpaceTimePointCloudNI, self).__init__()
        self.batchsize = batchsize
        self.fraction_on_surface = fraction_on_surface
        self.fraction_off_surface = fraction_off_surface
        self.device = device

        vertices = [None] * len(inputpaths)
        ni = [None] * len(inputpaths)
        faces = [None] * len(inputpaths)
        nf = [None] * len(inputpaths)
        self.kdtree = [None]*len(inputpaths)
        self.init_mesh = [None]*len(inputpaths)
        
        for i, (meshpath, nipath, paramval, w0) in enumerate(inputpaths):
            tri, verts = _read_ply(meshpath, paramval)

            ni[i] = from_pth(nipath, device=device, w0=w0, hd = hd).to(device)
            faces[i] = tri.to(device)
            nf[i] = compute_normales_faces(verts[:,0:3].numpy(), tri.numpy()).to(device)
            face_tri = verts[tri,:3]
            f_area = torch.cross(face_tri[...,0, :]-face_tri[..., 1, :], face_tri[...,0, :]-face_tri[..., 2, :], -1).norm(dim=-1)
            vertices[i] = self.mesh_sampler(100000,verts.detach().cpu().numpy(), tri.detach().cpu().numpy(), nf[i].detach().cpu().numpy(), f_area.detach().cpu().numpy())
            self.init_mesh[i] = verts.to(device)
                        
        self.vertices_ni = list(zip(vertices, ni, faces,nf,f_area))
        

        self.off_surf_sampler = torch.distributions.uniform.Uniform(-1*size_domain, 1*size_domain)
        self.time_sampler = torch.distributions.uniform.Uniform(something, 1.0)
 

    #Be careful, this is not an uniform mesh sampler !
    @torch.no_grad()
    def mesh_sampler(self,n,vert, faces, normales_faces, f_area):
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

        return torch.cat((torch.tensor(plus_pc).to(self.device), vert[0,3]*torch.ones(n,1,device=self.device), torch.tensor(plus_n).to(self.device), torch.zeros(n,1,device=self.device)), -1).to(torch.float32)
    
    def sample_edition(self, e) : 
        
        return _create_training_data(
            self.vertices_ni,
            n_samples=self.batchsize,
            fraction_on_surface=0.5,
            fraction_off_surface=0,
            off_surface_sampler=self.off_surf_sampler,
            time_sampler=self.time_sampler,
            device=self.device
        )
                
    
    def initialize_kdtree(self): 
        for i in range(2): 
            self.kdtree[i] = KDTree(self.vertices_ni[i][0][:,:3].cpu())

    def __len__(self):
        return sum([m.shape[0] for m, _, _, _, _ in self.vertices_ni])

    def __getitem__(self, n):
        return _create_training_data(
            self.vertices_ni,
            n_samples=self.batchsize,
            fraction_on_surface=self.fraction_on_surface,
            fraction_off_surface=self.fraction_off_surface,
            off_surface_sampler=self.off_surf_sampler,
            time_sampler=self.time_sampler,
            device=self.device
        )












