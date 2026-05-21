# coding: utf-8
import math

import numpy as np
import torch
from torch.autograd import grad


def outter(v1, v2):
    """
    Batched outter product of two vectors: [v1] [v2]^T
    :param v1: (bs, dim)
    :param v2: (bs, dim)
    :return: (bs, dim, dim)
    """
    bs = v1.size(0)
    d = v1.size(1)
    v1 = v1.view(bs, d, 1)
    v2 = v2.view(bs, 1, d)
    return torch.bmm(v1, v2)


def addr(mat, vec1, vec2, alpha=1., beta=1.):
    """
    Return
        alpha * outter(vec1, vec2) + beta * [mat]
    :param mat:  (bs, npoints, dim, dim)
    :param vec1: (bs, npoints, dim)
    :param vec2: (bs, npoints, dim)
    :param alpha: float
    :param beta: float
    :return:
    """
    npoints, dim =vec1.size(0), vec1.size(1)

    assert len(mat.size()) == 3
    outter_n = outter(vec1.view(npoints, dim), vec2.view(npoints, dim))
    outter_n = outter_n.view(npoints, dim, dim)
    out = alpha * outter_n + beta * mat.view(npoints, dim, dim)
    return out
    



def hessian_nfgp(y, x):
    """
    Hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, dim)
    return:
        shape (meta_batch_size, num_observations, dim, channels)
    """
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(y.shape[0],y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y,
                                   create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h.view(y.shape[0], x.shape[-1], x.shape[-1]), status


def laplace_nfgp(y, x, normalize=False, eps=0., return_grad=False):
    grad = gradient(y, x)
    if normalize:
        grad = grad / (grad.norm(dim=-1, keepdim=True) + eps)
    div = divergence(grad, x)

    if return_grad:
        return div, grad
    return div


def divergence_nfgp(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(
            y[..., i], x, torch.ones_like(y[..., i]),
            create_graph=True)[0][..., i:i+1]
    return div


def gradient_nfgp(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def jacobian_nfgp(y, x):
    """
    Jacobian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, dim)
    ret: shape (meta_batch_size, num_observations, channels, dim)
    """
    meta_batch_size, num_observations = y.shape[:2]
    # (meta_batch_size*num_points, 2, 2)
    jac = torch.zeros( y.shape[0], y.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[ :, i, :] = grad(
            y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status
    
def tangent_proj_mat(y, x, norm=True, eps=1e-6):
    """
    Compute the tangential projection matrix:
        P = I - n(x)n(x)^T
        where n(x) is the outward surface normal of x
    :param x: (bs, npts, dim) input points
    :param y: (bs, npts, 1) neural_field(x)
    :param norm: Whether normalize the surface normal vector
    :param eps: Numerical eps
    :return:
        [normals] (bs, npts, dim) The surface normal
        [normals_proj] (bs, npts, dim, dim) The projector matrices
    """
    npoints, dim = x.size(0), x.size(1)
    grad = gradient_nfgp(y, x)
    if norm:
        normals = (
                grad / (grad[:,:-1].norm(dim=-1, keepdim=True) + eps)
        ).view( npoints, dim)
    else:
        normals = grad.view(npoints, dim)
    normals_proj = addr(
        torch.eye(dim).view(1, dim, dim).expand(npoints, -1, -1).to(y),
        normals, normals, alpha=-1
    )
    return normals, normals_proj
