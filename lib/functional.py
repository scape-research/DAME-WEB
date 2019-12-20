import math
import pdb

import torch
import torch.nn.functional as F

# --------------------------------------
# pooling
# --------------------------------------

def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_max_pool2d(x, (1,1)) # alternative


def spoc(x):
    return F.avg_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_avg_pool2d(x, (1,1)) # alternative


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


def rmac(x, L=3, eps=1e-6):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w/2.0 - 1)

    b = (max(H, W)-w)/(steps-1)
    (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:  
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)

        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b) - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b) - wl2 # center coordinates
            
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:,:,(int(i_)+torch.Tensor(range(wl)).long()).tolist(),:]
                R = R[:,:,:,(int(j_)+torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt

    return v


def roipool(x, rpool, L=3, eps=1e-6):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w/2.0 - 1)

    b = (max(H, W)-w)/(steps-1)
    _, idx = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:  
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    vecs = []
    vecs.append(rpool(x).unsqueeze(1))

    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)

        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b).int() - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b).int() - wl2 # center coordinates
            
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                vecs.append(rpool(x.narrow(2,i_,wl).narrow(3,j_,wl)).unsqueeze(1))

    return torch.cat(vecs, dim=1)


# --------------------------------------
# normalization
# --------------------------------------

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

def powerlaw(x, eps=1e-6):
    x = x + self.eps
    return x.abs().sqrt().mul(x.sign())

# --------------------------------------
# loss
# --------------------------------------
import sys

def contrastive_loss(x, label, margin=0.7, eps=1e-6):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    y = 0.5*lbl*torch.pow(D,2) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)

    y = torch.sum(y)
    
    # print(x1.shape)
    # print(x2.shape)
    # print(label)
    # print(lbl)
    # 
    # # torch.Size([512, 6])
    # # torch.Size([512, 6])
    # # tensor([-1.,  1.,  0.,  0.,  0.,  0.,  0.], device='cuda:0')
    # # tensor([1., 0., 0., 0., 0., 0.], device='cuda:0')

    # sys.exit()
    return y


def contrastive_KLdiv_loss(x, label, margin=0.7, eps=1e-6):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    y = 0.5*lbl*torch.pow(D,2) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)

    y = torch.sum(y)
    
    # print(x1.shape)
    # print(x2.shape)
    # print(label)
    # print(lbl)
    # 
    # # torch.Size([512, 6])
    # # torch.Size([512, 6])
    # # tensor([-1.,  1.,  0.,  0.,  0.,  0.,  0.], device='cuda:0')
    # # tensor([1., 0., 0., 0., 0., 0.], device='cuda:0')
                             
    # sys.exit()
    
    D_KL = torch.zeros(D.shape[0])
    for k in range(0,D.shape[0]):
        a_vec = x1[:,k]
        b_vec = x2[:,k]
        kl_div1 = -torch.sum(a_vec*torch.log(b_vec/a_vec))
        kl_div2 = -torch.sum(b_vec*torch.log(a_vec/b_vec))
        D_KL[k] = 0.5*kl_div1+0.5*kl_div2

    margin_KL = 5
    y_KL = 0.5*lbl*D_KL + 0.5*(1-lbl)*torch.clamp(margin_KL-D_KL, min=0)

    y_KL = torch.sum(y_KL)


    y = y + 0.2*y_KL

    return y