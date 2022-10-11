import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
import time
import scipy
import scipy.io as sio
from scipy.sparse import csr_matrix, csc_matrix
import os
from torch_scatter import scatter_add


SAVE_MEMORY = True

def distance_GIH(V, T, t=1e-1):
    
    W,A = LBO_slim(V, T)
    grad,div,N = _grad_div(V,T)
    
    D = _geodesics_in_heat(grad,div,W[0],A,t)
    d = torch.diag(D)[:,None]
    
    #WARNIG: original D is not symmetric, it is symmetrized and shifted to have diagonal equal to zero
    D = (D + D.t()-d-d.t())/2
#     d = torch.min(D,dim=0)[0][:,None]
#     D = D-d.t()
    
    return D, grad, div, W, A, N


def LBO_slim(V, F):
    """
    Input:
      V: B x N x 3
      F: B x F x 3
    Outputs:
      C: B x F x 3 list of cotangents corresponding
        angles for triangles, columns correspond to edges 23,31,12
    """
    #tensor type and device
    device = V.device
    dtype = V.dtype
    
    indices_repeat = torch.stack([F, F, F], dim=2)

    # v1 is the list of first triangles B*F*3, v2 second and v3 third
    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())

    l1 = torch.sqrt(((v2 - v3) ** 2).sum(2))  # distance of edge 2-3 for every face B*F
    l2 = torch.sqrt(((v3 - v1) ** 2).sum(2))
    l3 = torch.sqrt(((v1 - v2) ** 2).sum(2))

    # semiperimieters
    sp = (l1 + l2 + l3) * 0.5

    # Heron's formula for area
    # A = torch.sqrt( sp * (sp-l1)*(sp-l2)*(sp-l3)).cuda()
    A = 0.5 * (torch.sum(torch.cross(v2 - v1, v3 - v2, dim=2) ** 2, dim=2) ** 0.5)  # VALIDATED

    # Theoreme d Al Kashi : c2 = a2 + b2 - 2ab cos(angle(ab))
    cot23 = (l1 ** 2 - l2 ** 2 - l3 ** 2) / (8 * A)
    cot31 = (l2 ** 2 - l3 ** 2 - l1 ** 2) / (8 * A)
    cot12 = (l3 ** 2 - l1 ** 2 - l2 ** 2) / (8 * A)

    batch_cot23 = cot23.view(-1)
    batch_cot31 = cot31.view(-1)
    batch_cot12 = cot12.view(-1)

    # proof page 98 http://www.cs.toronto.edu/~jacobson/images/alec-jacobson-thesis-2013-compressed.pdf
    # C = torch.stack([cot23, cot31, cot12], 2) / torch.unsqueeze(A, 2) / 8 # dim: [B x F x 3] cotangent of angle at vertex 1,2,3 correspondingly

    B = V.shape[0]
    num_vertices_full = V.shape[1]
    num_faces = F.shape[1]

    edges_23 = F[:, :, [1, 2]]
    edges_31 = F[:, :, [2, 0]]
    edges_12 = F[:, :, [0, 1]]

    batch_edges_23 = edges_23.view(-1, 2)
    batch_edges_31 = edges_31.view(-1, 2)
    batch_edges_12 = edges_12.view(-1, 2)

    W = torch.zeros(B, num_vertices_full, num_vertices_full, dtype=dtype, device=device)

    repeated_batch_idx_f = torch.arange(0, B).repeat(num_faces).reshape(num_faces, B).transpose(1, 0).contiguous().view(
        -1)  # [000...111...BBB...], number of repetitions is: num_faces
    repeated_batch_idx_v = torch.arange(0, B).repeat(num_vertices_full).reshape(num_vertices_full, B).transpose(1,
                                                                                                      0).contiguous().view(
        -1)  # [000...111...BBB...], number of repetitions is: num_vertices_full
    repeated_vertex_idx_b = torch.arange(0, num_vertices_full).repeat(B)

    W[repeated_batch_idx_f, batch_edges_23[:, 0], batch_edges_23[:, 1]] = batch_cot23
    W[repeated_batch_idx_f, batch_edges_31[:, 0], batch_edges_31[:, 1]] = batch_cot31
    W[repeated_batch_idx_f, batch_edges_12[:, 0], batch_edges_12[:, 1]] = batch_cot12

    W = W + W.transpose(2, 1)

    batch_rows_sum_W = torch.sum(W, dim=1).view(-1)
    W[repeated_batch_idx_v, repeated_vertex_idx_b, repeated_vertex_idx_b] = -batch_rows_sum_W
    # W is the contangent matrix VALIDATED
    # Warning: residual error of torch.max(torch.sum(W,dim = 1).view(-1)) is ~ 1e-18

    VF_adj = VF_adjacency_matrix(V[0], F[0]).unsqueeze(0).expand(B, num_vertices_full, num_faces)  # VALIDATED
    V_area = (torch.bmm(VF_adj, A.unsqueeze(2)) / 3).squeeze()  # VALIDATED

    return W, V_area


def _grad_div(V,T):
    
    #tensor type and device
    device = V.device
    dtype = V.dtype
    
    #WARNING not sure about this
    bs = V.shape[0]
    V = V.reshape([-1,3])
    T = T.reshape([-1,3])

    XF = V[T,:].transpose(0,1)

    Na = torch.cross(XF[1]-XF[0],XF[2]-XF[0])
    A = torch.sqrt(torch.sum(Na**2,-1,keepdim=True))+1e-6
    N = Na/A
    dA = 0.5/A

    m = T.shape[0]
    n = V.shape[0]
    
    def grad(f):
        gf = torch.zeros(m,3,f.shape[-1], device=device, dtype=dtype)
        for i in range(3):
            s = (i+1)%3
            t = (i+2)%3
            v = -torch.cross(XF[t]-XF[s],N)
            if SAVE_MEMORY:
                gf.add_(f[T[:,i],None,:]*(dA[:,0,None,None]*v[:,:,None])) #Slower less-memeory
            else:
                gf.add_(f[T[:,i],None,:]*(dA[:,0,None,None]*v[:,:,None])) 
        return gf
    
    def div(f):
        gf = torch.zeros(f.shape[-1],n, device=device, dtype=dtype)        
        for i in range(3):
            s = (i+1)%3
            t = (i+2)%3
            v = torch.cross(XF[t]-XF[s],N)
            if SAVE_MEMORY:
                gf.add_(scatter_add( torch.bmm(v[:,None,:],f)[:,0,:].t(), T[:,i], dim_size=n))# slower but uses less memory
            else:
                gf.add_(scatter_add( (f*v[:,:,None]).sum(1).t(), T[:,i], dim_size=n))
        return gf.t()
    
#     W = div(grad(torch.eye(n).cuda().double()))
#     A = scatter_add(A[:,0],T[:,0]).scatter_add(0,T[:,1],A[:,0]).scatter_add(0,T[:,2],A[:,0])/6
    return grad, div, A


def _geodesics_in_heat(grad, div, W, A, t=1e-1):
    
    
    nsplits=1
    if SAVE_MEMORY:
        nsplits=5
        
    #tensor type and device
    device = W.device
    dtype = W.dtype
    
    n = W.shape[0]  
    n_chunk = int(n/nsplits)
    D = torch.zeros(n,  n, dtype=dtype, device=device)
    
    B = torch.diag(A) + t * W
    
    for i in range(nsplits):
        i1 = i*n_chunk
        i2 = np.min([n,(i+1)*n_chunk]).item()

        #U = torch.eye(n, dtype=dtype, device=device)
        U = torch.zeros(n, i2 - i1, dtype=dtype, device=device)
        U[i1:i2, :(i2 - i1)] = torch.eye((i2 - i1), dtype=dtype, device=device)
        f = torch.linalg.solve(B, U)#[0]
        gf = grad(f)
        gf = gf*(gf.pow(2).sum(1,keepdims=True)+1e-12).rsqrt()
        
        Di = torch.linalg.solve(W, div(gf))#[0]
        D[:,i1:i2] = Di
    return D


def VF_adjacency_matrix(V, F):
    """
    Input:
    V: N x 3
    F: F x 3
    Outputs:
    C: V x F adjacency matrix
    """
    #tensor type and device
    device = V.device
    dtype = V.dtype
    
    VF_adj = torch.zeros((V.shape[0], F.shape[0]), dtype=dtype, device=device)
    v_idx = F.view(-1)
    f_idx = torch.arange(F.shape[0]).repeat(3).reshape(3, F.shape[0]).transpose(1, 0).contiguous().view(
        -1)  # [000111...FFF]

    VF_adj[v_idx, f_idx] = 1
    return VF_adj

def calc_euclidean_dist_matrix(x):
    #OH: x contains the coordinates of the mesh,
    #x dimensions are [batch_size x num_nodes x 3]

    #x = x.transpose(2,1)
    r = torch.sum(x ** 2, dim=2).unsqueeze(2)  # OH: [batch_size  x num_points x 1]
    r_t = r.transpose(2, 1) # OH: [batch_size x 1 x num_points]
    inner = torch.bmm(x,x.transpose(2, 1))
    D = F.relu(r - 2 * inner + r_t)**0.5  # OH: the residual numerical error can be negative ~1e-16
    return D