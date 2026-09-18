from __future__ import annotations
import os, time, math, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- configuration ----------------
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(min(8, os.cpu_count() or 4))
DEVICE = torch.device('cpu')
DT = 0.1
NTRAIN_T = 96
NTOTAL = 192
ALPHA_TRUE = 0.65
THETA_TRUE = 0.12
C_TRUE = 1.0
RHO_MAIN = 0.90
N_TRAIN, N_VAL, N_TEST = 96, 24, 32
MAIN_SEEDS = [11, 22]
NEAR_RHOS = [0.70, 0.90, 0.97]
NEAR_SEED = 44
EPOCHS = {
    'fLNO': 100,
    'PosRationalLNO': 110,
    'UnconstrainedRational': 110,
    'FNO1D': 220,
    'DeepONet': 600,
    'LinearSSM': 110,
}
BATCH=24

# ---------------- reproducibility ----------------
def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def rel_l2(yh, y):
    num = torch.linalg.vector_norm((yh-y).reshape(y.shape[0], -1), dim=1)
    den = torch.linalg.vector_norm(y.reshape(y.shape[0], -1), dim=1).clamp_min(1e-8)
    return (num/den).mean().item()

def sup_rel(yh,y):
    num=(yh-y).abs().amax(dim=1)
    den=y.abs().amax(dim=1).clamp_min(1e-8)
    return (num/den).mean().item()

def tail_rel(yh,y, start=NTRAIN_T):
    return rel_l2(yh[:,start:], y[:,start:])

# ---------------- forcing generation ----------------
def make_forcings(n, nt=NTOTAL, active=NTRAIN_T, seed=0):
    rng=np.random.default_rng(seed)
    t=np.arange(nt)*DT
    U=np.zeros((n,nt),dtype=np.float32)
    for i in range(n):
        u=np.zeros(nt)
        # low-frequency Fourier mixture
        nf=rng.integers(2,5)
        for _ in range(nf):
            freq=rng.uniform(0.025,0.18)
            amp=rng.normal(0,0.28)
            phase=rng.uniform(0,2*np.pi)
            u += amp*np.sin(2*np.pi*freq*t+phase)
        # smooth Gaussian pulses
        for _ in range(rng.integers(1,4)):
            center=rng.uniform(1.0, active*DT-1.0)
            width=rng.uniform(0.25,1.2)
            amp=rng.normal(0,0.45)
            u += amp*np.exp(-0.5*((t-center)/width)**2)
        # piecewise bias over active interval
        cuts=sorted(rng.choice(np.arange(8,active-8),size=2,replace=False))
        vals=rng.normal(0,0.12,3)
        u[:cuts[0]] += vals[0]; u[cuts[0]:cuts[1]] += vals[1]; u[cuts[1]:active] += vals[2]
        # zero future forcing: genuine autonomous extrapolation
        u[active:]=0.0
        # bounded scale
        mx=max(np.max(np.abs(u[:active])),1e-6)
        u[:active] *= min(1.0,0.9/mx)
        U[i]=u.astype(np.float32)
    return U

# ---------------- fractional ground-truth solver ----------------
def frac_weights_torch(alpha, theta, n, dt=DT, dtype=torch.float32):
    k=torch.arange(1,n+1,device=DEVICE,dtype=dtype)
    tm=(k-0.5)*dt
    logg=-theta*tm-alpha*torch.log(tm)-torch.lgamma(1-alpha)
    return dt*torch.exp(logg)

def frac_solve_torch(u, alpha, theta, rho, c=1.0, nt=None):
    # u: B,T. Midpoint quadrature for g(t)=exp(-theta t)t^-alpha/Gamma(1-alpha)
    if nt is None: nt=u.shape[1]
    u=u[:,:nt]
    Gmass=theta.pow(alpha-1)
    a=rho*c/Gmass
    w=frac_weights_torch(alpha,theta,nt,DT,u.dtype)
    ys=[]
    yprev=torch.zeros(u.shape[0],device=u.device,dtype=u.dtype)
    denom=1.0/DT+c
    for n in range(nt):
        if n==0:
            mem=torch.zeros_like(yprev)
        else:
            hist=torch.stack(ys,dim=1)  # B,n
            mem=(hist*torch.flip(w[:n],dims=[0])[None,:]).sum(dim=1)
        yn=(yprev/DT + a*mem + u[:,n])/denom
        ys.append(yn); yprev=yn
    return torch.stack(ys,dim=1)

def generate_dataset(rho, seed):
    U=make_forcings(N_TRAIN+N_VAL+N_TEST,seed=seed)
    with torch.no_grad():
        ut=torch.tensor(U,device=DEVICE)
        y=frac_solve_torch(ut,torch.tensor(ALPHA_TRUE),torch.tensor(THETA_TRUE),torch.tensor(rho),C_TRUE,NTOTAL).cpu().numpy().astype(np.float32)
    return U,y

# ---------------- models ----------------
class fLNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_alpha=nn.Parameter(torch.tensor(0.35))
        self.raw_theta=nn.Parameter(torch.tensor(math.log(math.exp(0.2)-1)))
        self.raw_rho=nn.Parameter(torch.tensor(2.0))
    def pars(self):
        alpha=0.05+0.90*torch.sigmoid(self.raw_alpha)
        theta=0.015+F.softplus(self.raw_theta)
        rho=0.999*torch.sigmoid(self.raw_rho)
        return alpha,theta,rho
    def forward(self,u,nt=None):
        a,t,r=self.pars()
        return frac_solve_torch(u,a,t,r,torch.tensor(C_TRUE,device=u.device),nt)
    def margin(self):
        return float(1-self.pars()[2].detach())
    def extra(self):
        a,t,r=self.pars(); return {'alpha_hat':float(a.detach()),'theta_hat':float(t.detach()),'rho_hat_param':float(r.detach())}

class PositiveRationalLNO(nn.Module):
    def __init__(self,K=12):
        super().__init__(); self.K=K
        # log-spaced initialization over memory scales
        rates=np.geomspace(0.05,6.0,K).astype(np.float32)
        self.raw_rates=nn.Parameter(torch.tensor(np.log(np.expm1(rates))))
        self.raw_pi=nn.Parameter(torch.zeros(K))
        self.raw_rho=nn.Parameter(torch.tensor(2.0))
    def pars(self):
        rates=0.01+F.softplus(self.raw_rates)
        pi=torch.softmax(self.raw_pi,dim=0)
        rho=0.999*torch.sigmoid(self.raw_rho)
        q=rho*C_TRUE*pi*rates  # sum q/r = rho*c
        return rates,q,rho
    def forward(self,u,nt=None):
        if nt is None: nt=u.shape[1]
        rates,q,rho=self.pars(); B=u.shape[0]
        z=torch.zeros(B,self.K,device=u.device); yprev=torch.zeros(B,device=u.device)
        decay=torch.exp(-rates*DT); integ=(1-decay)/rates
        ys=[]; denom=1/DT+C_TRUE
        for n in range(nt):
            z=decay[None,:]*z + integ[None,:]*yprev[:,None]
            yn=(yprev/DT + (z*q[None,:]).sum(1)+u[:,n])/denom
            ys.append(yn); yprev=yn
        return torch.stack(ys,1)
    def margin(self): return float(1-self.pars()[2].detach())
    def extra(self): return {'rho_hat_param':float(self.pars()[2].detach())}

class UnconstrainedRational(nn.Module):
    def __init__(self,K=12):
        super().__init__(); self.K=K
        rates=np.geomspace(0.05,6.0,K).astype(np.float32)
        self.raw_rates=nn.Parameter(torch.tensor(np.log(np.expm1(rates))))
        self.q=nn.Parameter(torch.zeros(K))
        # initialize near positive stable kernel
        with torch.no_grad(): self.q[:] = torch.tensor(0.75*C_TRUE*np.ones(K)/K*rates,dtype=torch.float32)
    def pars(self): return 0.01+F.softplus(self.raw_rates), self.q
    def forward(self,u,nt=None):
        if nt is None: nt=u.shape[1]
        rates,q=self.pars(); B=u.shape[0]
        z=torch.zeros(B,self.K,device=u.device); yprev=torch.zeros(B,device=u.device)
        decay=torch.exp(-rates*DT); integ=(1-decay)/rates
        ys=[]; denom=1/DT+C_TRUE
        for n in range(nt):
            z=decay[None,:]*z + integ[None,:]*yprev[:,None]
            yn=(yprev/DT + (z*q[None,:]).sum(1)+u[:,n])/denom
            ys.append(yn); yprev=yn
        return torch.stack(ys,1)
    def characteristic_roots(self):
        rates,q=self.pars(); r=rates.detach().cpu().numpy(); q=q.detach().cpu().numpy()
        # P(s)=(s+c) prod(s+r_k) - sum q_k prod_{j != k}(s+r_j)
        prod=np.poly1d([1.0])
        for rk in r: prod=np.polymul(prod,np.poly1d([1.0,rk]))
        P=np.polymul(np.poly1d([1.0,C_TRUE]²È="25Ñ}±…å½ÕĞ ¤ì™¥œ¹Í…Ù•™¥œ¡=UP¼‰•¹¡µ…É­}•á…µÁ±•}ÑÉ…©•Ñ½É¥•Ì¹Á¹œœ±‘Á¤ôÈÈÀ¤ìÁ±Ğ¹±½Í”¡™¥œ¤(€€€É•ÑÕÉ¸‘˜((Œ€´´´´´´´´´´´´´´´´¹•…ÈµÉ¥Ñ¥…°Íİ••À€´´´´´´´´´´´´´´´´)‘•˜ÉÕ¹}¹•…È ¤è(€€€É½İÌõmt(€€€™½ÈÉ¡¼¥¸9I}I!=Lè(€€€€€€€ÁÉ¥¹Ğ¡˜q¸ôôô9HµI%Q%0É¡¼õíÉ¡¼è¸É™ô€ôôôœ±™±ÕÍ õQÉÕ”¤(€€€€€€€T±dõ•¹•É…Ñ•}‘…Ñ…Í•Ğ¡É¡¼°ÔÀÀÀ­¥¹Ğ¡É¡¼¨ÄÀÀ¤¤(€€€€€€€™½È¹…µ”¥¸5=1Lè(€€€€€€€€€€€€ŒÍ±¥¡Ñ±äÉ•‘Õ•‰ÕĞÍÑ¥±°ÍÕ‰ÍÑ…¹Ñ¥Ù”ÑÉ…¥¹¥¹œ(€€€€€€€€€€€”õµ…à äÀ±¥¹Ğ¡A=!Mm¹…µ•t¨À¸ØÔ¤¤(€€€€€€€€€€€µ½‘•°±Á˜±ÑĞ±|õÑÉ…¥¹}µ½‘•°¡¹…µ”±9I}M­¥¹Ğ¡É¡¼¨ÄÀÀ¤±T±d±•Á½¡Ìõ”¤(€€€€€€€€€€€É½Ü±|±|õ•Ù…±}µ½‘•°¡¹…µ”±µ½‘•°±Á˜±T±d±É¡¼±ÑĞ¤(€€€€€€€€€€€É½İlÍ••tõ9I}MìÉ½İlÁ¡…Í”tô¹•…ÈœìÉ½İÌ¹…ÁÁ•¹¡É½Ü¤(€€€€€€€€€€€ÁÉ¥¹Ğ¡¹…µ”°Ñ…¥°œ±É½Õ¹¡É½İlÑ…¥±}É•±0Èt°Ğ¤°µ…É¥¹}•ÉÈœ±É½Õ¹¡É½İl‰É…¹¡}µ…É¥¹}…‰Í}•ÉÉ½Èt°Ğ¤±™±ÕÍ õQÉÕ”¤(€€€‘˜õÁ¹…Ñ…É…µ”¡É½İÌ¤ì‘˜¹Ñ½}ÍØ¡=UP¼¹•…É}É¥Ñ¥…±}É•ÍÕ±ÑÌ¹ÍØœ±¥¹‘•àõ…±Í”¤(€€€™¥œ±…àõÁ±Ğ¹ÍÕ‰Á±½ÑÌ¡™¥Í¥é”ô Ü¸È°Ğ¸Ô¤¤(€€€™½È¹…µ”±œ¥¸‘˜¹É½ÕÁ‰ä µ½‘•°œ¤è(€€€€€€€…à¹Á±½Ğ¡lµ…É¥¹}ÑÉÕ”t±lÑ…¥±}É•±0Èt°¼´œ±±…‰•°õ¹…µ”¤(€€€…à¹Í•Ñ}áÍ…±” ±½œœ¤ì…à¹Í•Ñ}åÍ…±” ±½œœ¤ì…à¹¥¹Ù•ÉÑ}á…á¥Ì ¤ì…à¹Í•Ñ}á±…‰•° ÑÉÕ”ÍÑ…‰¥±¥Ñäµ…É¥¸€ÄµÉ¡¼œ¤ì…à¹Í•Ñ}å±…‰•° œÉàµ¡½É¥é½¸Ñ…¥°É•±…Ñ¥Ù”0Èœ¤(€€€…à¹É¥¡…±Á¡„ô¸ÈÔ±İ¡¥ ô‰½Ñ œ¤ì…à¹±••¹¡™½¹ÑÍ¥é”ôÜ±¹½°ôÈ¤ì™¥œ¹Ñ¥¡Ñ}±…å½ÕĞ ¤ì™¥œ¹Í…Ù•™¥œ¡=UP¼¹•…É}É¥Ñ¥…±}Íİ••À¹Á¹œœ±‘Á¤ôÈÈÀ¤ìÁ±Ğ¹±½Í”¡™¥œ¤(€€€É•ÑÕÉ¸‘˜((Œ€´´´´´´´´´´´´´´´´É•Á•…Ñ•ÍÑ…‰¥±¥Ñä…Õ‘¥Ğ€´´´´´´´´´´´´´´´´)‘•˜ÉÕ¹}ÍÑ…‰¥±¥Ñå}…Õ‘¥Ğ¡¹Í••‘Ìôà¤è(€€€É½İÌõmtìÉ¡¼ôÀ¸äÜìT±dõ•¹•É…Ñ•}‘…Ñ…Í•Ğ¡É¡¼°àÀàÀ¤(€€€™½ÈÌ¥¸É…¹”¡¹Í••‘Ì¤è(€€€€€€€Í••ôÜÀÀ­Ì(€€€€€€€™½È¹…µ”¥¸l™19<œ°A½ÍI…Ñ¥½¹…±19<œ°U¹½¹ÍÑÉ…¥¹•‘I…Ñ¥½¹…°œ°1¥¹•…ÉMM4œ°9<Åœ°••Á=9•Ğtè(€€€€€€€€€€€”ôÄÀÀ¥˜¹…µ”¹½Ğ¥¸ì••Á=9•Ğô•±Í”€ÄÈÀ(€€€€€€€€€€€µ½‘•°±Á˜±ÑĞ±|õÑÉ…¥¹}µ½‘•°¡¹…µ”±Í••±T±d±•Á½¡Ìõ”¤(€€€€€€€€€€€É½Ü±|±|õ•Ù…±}µ½‘•°¡¹…µ”±µ½‘•°±Á˜±T±d±É¡¼±ÑĞ¤(€€€€€€€€€€€€Œ•á…ĞÕ¹ÍÑ…‰±”™±…œİ¡•É”‘•™¥¹•(€€€€€€€€€€€¥˜¹…µ”ôôU¹½¹ÍÑÉ…¥¹•‘I…Ñ¥½¹…°œèÕ¹ÍÑ…‰±”õ™±½…Ğ¡É½Ü¹•Ğ µ…á}É½½Ñ}É•…°œ°´Ä¤øôÀ¤(€€€€€€€€€€€•±¥˜¹…µ”¥¸ì™19<œ°A½ÍI…Ñ¥½¹…±19<œ°1¥¹•…ÉMM4ôèÕ¹ÍÑ…‰±”ôÀ¸À(€€€€€€€€€€€•±Í”èÕ¹ÍÑ…‰±”õ¹À¹¹…¸(€€€€€€€€€€€É½İÌ¹…ÁÁ•¹¡ìµ½‘•°œé¹…µ”°Í••œéÍ••°ÍÑÉÕÑÕÉ…±}Õ¹ÍÑ…‰±”œéÕ¹ÍÑ…‰±”°(€€€€€€€€€€€€€€€€€€€€€€€€€•µÁ¥É¥…±}‰±½İÕÀœéÉ½İl•µÁ¥É¥…±}‰±½İÕÁ}É…Ñ”t°(€€€€€€€€€€€€€€€€€€€€€€€€€Ñ…¥±}É•±0ÈœéÉ½İlÑ…¥±}É•±0Èt°(€€€€€€€€€€€€€€€€€€€€€€€€€É¡½}¡…Ñ}ÍÑ•ÀœéÉ½İlÉ¡½}¡…Ñ}ÍÑ•Àt°(€€€€€€€€€€€€€€€€€€€€€€€€€µ…á}É½½Ñ}É•…°œéÉ½Ü¹•Ğ µ…á}É½½Ñ}É•…°œ±¹À¹¹…¸¥ô¤(€€€€€€€€€€€ÁÉ¥¹Ğ …Õ‘¥Ğœ±Í••±¹…µ”°Õ¹ÍÑ…‰±”œ±Õ¹ÍÑ…‰±”°‰±½Üœ±É½İl•µÁ¥É¥…±}‰±½İÕÁ}É…Ñ”t±™±ÕÍ õQÉÕ”¤(€€€‘˜õÁ¹…Ñ…É…µ”¡É½İÌ¤ì‘˜¹Ñ½}ÍØ¡=UP¼ÍÑ…‰¥±¥Ñå}…Õ‘¥Ğ¹ÍØœ±¥¹‘•àõ…±Í”¤(€€€É•ÑÕÉ¸‘˜((Œ€´´´´´´´´´´´´´´´´É…Á µÍ¥é”ÑÉ…¹Í™•È™½ÈÍ¥é”µ¥¹‘•Á•¹‘•¹Ğ„¡±…µ‰‘„¤€´´´´´´´´´´´´´´´´)‘•˜É…Á¡}ÑÉ…¹Í™•É}•áÁ•É¥µ•¹Ğ¡Í••ôäÄ¤è(€€€€Œ‘•Ñ•Éµ¥¹¥ÍÑ¥ŒÉ¥¹œµ±¥­”İ•¥¡Ñ•É…Á¡Ìİ¥Ñ ¹½Éµ…±¥é•1…Á±…¥…¸•¥•¹Ù…±Õ•Ì¥¸lÀ°Ét(€€€€ŒÑÉÕ”µ½‘…°…¥¸„¡±…µ‰‘„¤õ­…ÁÁ„¼ Ä­‰•Ñ„©±…µ‰‘„¤ì±•…É¸Ñ¡•Ñ„ô¡…±Á¡„±Ñ¡•Ñ„±­…ÁÁ„±‰•Ñ„¤…Ğ¸ôĞà°‘•Á±½ä¸ôäØ°ÄäÈ¸(€€€Í••‘}…±°¡Í••¤(€€€É¹œõ¹À¹É…¹‘½´¹‘•™…Õ±Ñ}É¹œ¡Í••¤(€€€…±Á¡„±Ñ¡•Ñ„±­…ÁÁ„±‰•Ñ„ôÀ¸ØÔ°À¸ÄØ°À¸ĞÈ°À¸ÜÔ(€€€ŒôÄ¸À(€€€‘•˜É¥¹}±…Á±…¥…¸¡¸¤è(€€€€€€€õ¹À¹é•É½Ì ¡¸±¸¤¤(€€€€€€€™½È¤¥¸É…¹”¡¸¤è(€€€€€€€€€€€m¤°¡¤´Ä¤•¹tõm¤°¡¤¬Ä¤•¹tôÄ¸À(€€€€€€€€€€€m¤°¡¤´È¤•¹tõm¤°¡¤¬È¤•¹tôÀ¸ÌÔ(€€€€€€€õ¹ÍÕ´ Ä¤ìõ¹À¹‘¥…œ¡¤ì0õµ(€€€€€€€€Œ¹½Éµ…±¥é•‰äµ…à‘•É•”(€€€€€€€É•ÑÕÉ¸0½¹µ…à ¤(€€€‘•˜…¥¹Ì¡•¥Ì±­…À±‰•Ğ¤èÉ•ÑÕÉ¸­…À¼ Ä­‰•Ğ©•¥Ì¤(€€€‘•˜Í½±Ù•}µ½‘•Í}¹À¡•¥Ì°´°¹Ğ°Á…ÉÌ¤è(€€€€€€€…°±Ñ ±­…À±‰•ĞõÁ…ÉÌìõÑ ¨¨¡…°´Ä¤ì„õ…¥¹Ì¡•¥Ì±­…À±‰•Ğ¤ì€ŒµÕÍĞ‰”ÍÕ‰É¥Ñ¥…°„©ñŒ(€€€€€€€€ŒÑ½É Ù•Ñ½É¥é•Á•Èµ½‘•Ì…Ì‰…Ñ (€€€€€€€ÕĞõÑ½É ¹Ñ•¹Í½È¡´¹P¹…ÍÑåÁ”¡¹À¹™±½…ĞÌÈ¤¤€Œµ½‘•ÌàÑ¥µ”(€€€€€€€€ŒÍ½±Ù”•… µ½‘”İ¥Ñ ¥ÑÌ½İ¸É¡¼õ…½ŒèÉ•Á±¥…Ñ”Í½±Ù•Èµ…¹Õ…±±äİ¥Ñ Á•Èµ‰…Ñ É¡¼(€€€€€€€…±PõÑ½É ¹Ñ•¹Í½È¡…°¤ìÑ¡PõÑ½É ¹Ñ•¹Í½È¡Ñ ¤ìÜõ™É…}İ•¥¡ÑÍ}Ñ½É ¡…±P±Ñ¡P±¹Ğ±P¤(€€€€€€€…PõÑ½É ¹Ñ•¹Í½È¡„¹…ÍÑåÁ”¡¹À¹™±½…ĞÌÈ¤¤ìåÌõmtìåÀõÑ½É ¹é•É½Ì¡±•¸¡•¥Ì¤¤ì‘•¸ôÄ½P­Œ(€€€€€€€™½È¸¥¸É…¹”¡¹Ğ¤è(€€€€€€€€€€€¥˜¸ôôÀèµ•´õÑ½É ¹é•É½Í}±¥­”¡åÀ¤(€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€¡¥ÍĞõÑ½É ¹ÍÑ…¬¡åÌ°Ä¤ìµ•´ô¡¡¥ÍĞ©Ñ½É ¹™±¥À¡İlé¹t±lÁt¥m9½¹”°ét¤¹ÍÕ´ Ä¤(€€€€€€€€€€€å¸ô¡åÀ½P­…P©µ•´­ÕÑlè±¹t¤½‘•¸ìåÌ¹…ÁÁ•¹¡å¸¤ìåÀõå¸(€€€€€€€É•ÑÕÉ¸Ñ½É ¹ÍÑ…¬¡åÌ°Ä¤¹P¹¹ÕµÁä ¤€ŒÑ¥µ”±µ½‘•Ì(€€€¸ôĞàì0õÉ¥¹}±…Á±…¥…¸¡¸¤ì•¥œ±Xõ¹À¹±¥¹…±œ¹•¥ ¡0¤(€€€€ŒÑÉ…¥¸½¸€Ô±½Üµ™É•ÅÕ•¹äµ½‘…°™½É¥¹Ì°‘¥É•Ğ±•…ÍĞÍÅÕ…É•Ì½Ù•ÈÁ¡åÍ¥…°Á…É…µÌİ¥Ñ Ñ½É ½ÁÑ¥µ¥é•È(€€€¹ĞôäØìµ…ÑÌõmtìdõmt(€€€Ğõ¹À¹…É…¹”¡¹Ğ¤©P(€€€™½ÈÄ¥¸É…¹” Ô¤è(€€€€€€€´õ¹À¹é•É½Ì ¡¹Ğ±¸¤±‘ÑåÁ”õ¹À¹™±½…ĞÌÈ¤(€€€€€€€™½È¨¥¸É…¹” à¤èµlè±©tôÀ¸Ì©¹À¹Í¥¸  À¸ÄÈ¬À¸ÀÌ©¨¤©Ğ­É¹œ¹Õ¹¥™½É´ À°Ø¸Èà¤¤(€€€€€€€µ…ÑÌ¹…ÁÁ•¹¡´¤ìd¹…ÁÁ•¹¡Í½±Ù•}µ½‘•Í}¹À¡•¥œ±´±¹Ğ°¡…±Á¡„±Ñ¡•Ñ„±­…ÁÁ„±‰•Ñ„¤¤¤(€€€€Œ½ÁÑ¥µ¥é”…±Á¡„Ñ¡•Ñ„­…ÁÁ„‰•Ñ„°•á…Ğµ½‘…°Í½±Ù•È¥¸Ñ½É İ¥Ñ Á•Èµµ½‘”…¥¹Ì(€€€É…Üõ¹¸¹A…É…µ•Ñ•È¡Ñ½É ¹Ñ•¹Í½È¡lÀ¸Ì°´Ä¸à°´À¸Ô°À¸Át¤¤ì½ÁĞõÑ½É ¹½ÁÑ¥´¹‘…´¡mÉ…İt±±Èô¸ÀĞ¤(€€€ÑÌõmÑ½É ¹Ñ•¹Í½È¡à¤™½Èà¥¸µ…ÑÍtìeÑÌõmÑ½É ¹Ñ•¹Í½È¡à¤™½Èà¥¸et(€€€™½È•À¥¸É…¹” ÄÈÀ¤è(€€€€€€€…°ô¸ÀÔ¬¸ä©Ñ½É ¹Í¥µ½¥¡É…İlÁt¤ìÑ ô¸ÀÈ­¹Í½™ÑÁ±ÕÌ¡É…İlÅt¤ì­…Àô¸ÀÔ­¹Í½™ÑÁ±ÕÌ¡É…İlÉt¤ì‰•Ğô¸ÀÔ­¹Í½™ÑÁ±ÕÌ¡É…İlÍt¤(€€€€€€€Üõ™É…}İ•¥¡ÑÍ}Ñ½É ¡…°±Ñ ±¹Ğ±P¤ìœõ­…À¼ Ä­‰•Ğ©Ñ½É ¹Ñ•¹Í½È¡•¥œ±‘ÑåÁ”õÑ½É ¹™±½…ĞÌÈ¤¤ì±½ÍÌôÀ(€€€€€€€™½ÈÔ±eÔ¥¸é¥À¡ÑÌ±eÑÌ¤è(€€€€€€€€€€€åÀõÑ½É ¹é•É½Ì¡¸¤ìåÌõmtìÁÉ•õmt(€€€€€€€€€€€™½ÈÑ¤¥¸É…¹”¡¹Ğ¤è(€€€€€€€€€€€€€€€¥˜Ñ¤ôôÀèµ•´õÑ½É ¹é•É½Í}±¥­”¡åÀ¤(€€€€€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€€€€€¡¥ÍĞõÑ½É ¹ÍÑ…¬¡åÌ°Ä¤ìµ•´ô¡¡¥ÍĞ©Ñ½É ¹™±¥À¡İléÑ¥t±lÁt¥m9½¹”°ét¤¹ÍÕ´ Ä¤(€€€€€€€€€€€€€€€å¸ô¡åÀ½P­œ©µ•´­ÕmÑ¥t¤¼€ Ä½P­Œ¤ìåÌ¹…ÁÁ•¹¡å¸¤ìÁÉ•¹…ÁÁ•¹¡å¸¤ìåÀõå¸(€€€€€€€€€€€@õÑ½É ¹ÍÑ…¬¡ÁÉ•°À¤ì±½ÍÌõ±½ÍÌ­¹µÍ•}±½ÍÌ¡@±eÔ¤(€€€€€€€±½ÍÌõ±½ÍÌ½±•¸¡ÑÌ¤ì½ÁĞ¹é•É½}É… ¤ì±½ÍÌ¹‰…­İ…É ¤ì½ÁĞ¹ÍÑ•À ¤(€€€İ¥Ñ Ñ½É ¹¹½}É… ¤è(€€€€€€€Á ô¡™±½…Ğ ¸ÀÔ¬¸ä©Ñ½É ¹Í¥µ½¥¡É…İlÁt¤¤±™±½…Ğ ¸ÀÈ­¹Í½™ÑÁ±ÕÌ¡É…İlÅt¤¤±™±½…Ğ ¸ÀÔ­¹Í½™ÑÁ±ÕÌ¡É…İlÉt¤¤±™±½…Ğ ¸ÀÔ­¹Í½™ÑÁ±ÕÌ¡É…İlÍt¤¤¤(€€€É½İÌõmt(€€€™½È´¥¸lĞà°äØ°ÄäÉtè(€€€€€€€1´õÉ¥¹}±…Á±…¥…¸¡´¤ì•´±Y´õ¹À¹±¥¹…±œ¹•¥ ¡1´¤ìÑĞõ¹À¹…É…¹”¡¹Ğ¤©P(€€€€€€€´õ¹À¹é•É½Ì ¡¹Ğ±´¤±‘ÑåÁ”õ¹À¹™±½…ĞÌÈ¤(€€€€€€€™½È¨¥¸É…¹”¡µ¥¸ ÄÈ±´¤¤èµlè±©tôÀ¸ÌÔ©¹À¹½Ì  ¸ÄÀ¬¸ÀÈ©¨¤©ÑĞ¬¸Ì©¨¤(€€€€€€€åĞõÍ½±Ù•}µ½‘•Í}¹À¡•´±´±¹Ğ°¡…±Á¡„±Ñ¡•Ñ„±­…ÁÁ„±‰•Ñ„¤¤ìåÀõÍ½±Ù•}µ½‘•Í}¹À¡•´±´±¹Ğ±Á ¤(€€€€€€€•ÉÈõ¹À¹±¥¹…±œ¹¹½É´¡åÀµåĞ¤½¹À¹±¥¹…±œ¹¹½É´¡åĞ¤(€€€€€€€É½İÌ¹…ÁÁ•¹¡ì¸œé´°É•±0Èœé•ÉÈ°…±Á¡…}¡…ĞœéÁ¡lÁt°Ñ¡•Ñ…}¡…ĞœéÁ¡lÅt°­…ÁÁ…}¡…ĞœéÁ¡lÉt°‰•Ñ…}¡…ĞœéÁ¡lÍuô¤(€€€‘˜õÁ¹…Ñ…É…µ”¡É½İÌ¤ì‘˜¹Ñ½}ÍØ¡=UP¼É…Á¡}ÑÉ…¹Í™•É}É•ÍÕ±ÑÌ¹ÍØœ±¥¹‘•àõ…±Í”¤(€€€İ¥Ñ ½Á•¸¡=UP¼É…Á¡}ÑÉ…¹Í™•É}Á…É…µÌ¹©Í½¸œ°Üœ¤…Ì˜è©Í½¸¹‘ÕµÀ¡ìÑÉÕ”œém…±Á¡„±Ñ¡•Ñ„±­…ÁÁ„±‰•Ñ…t°±•…É¹•œé±¥ÍĞ¡Á ¥ô±˜±¥¹‘•¹ĞôÈ¤(€€€É•ÑÕÉ¸‘˜()¥˜}}¹…µ•}|ôô}}µ…¥¹}|œè(€€€¥µÁ½ÉĞ…ÉÁ…ÉÍ”(€€€…Àõ…ÉÁ…ÉÍ”¹ÉÕµ•¹ÑA…ÉÍ•È ¤ì…À¹…‘‘}…ÉÕµ•¹Ğ œ´µÁ¡…Í”œ±¡½¥•Ìõlµ…¥¸œ°¹•…Èœ°…Õ‘¥Ğœ°É…Á œ°…±°t±‘•™…Õ±Ğô…±°œ¤ì…ÉÌõ…À¹Á…ÉÍ•}…ÉÌ ¤(€€€ĞÀõÑ¥µ”¹Ñ¥µ” ¤(€€€¥˜…ÉÌ¹Á¡…Í”¥¸ìµ…¥¸œ°…±°ôèÉÕ¹}µ…¥¸ ¤(€€€¥˜…ÉÌ¹Á¡…Í”¥¸ì¹•…Èœ°…±°ôèÉÕ¹}¹•…È ¤(€€€¥˜…ÉÌ¹Á¡…Í”¥¸ì…Õ‘¥Ğœ°…±°ôèÉÕ¹}ÍÑ…‰¥±¥Ñå}…Õ‘¥Ğ Ğ¤(€€€¥˜…ÉÌ¹Á¡…Í”¥¸ìÉ…Á œ°…±°ôèÉ…Á¡}ÑÉ…¹Í™•É}•áÁ•É¥µ•¹Ğ ¤(€€€µ•Ñ„õì‘ĞœéP°9}ÑÉ…¥¹}Ñ¥µ”œé9QI%9}P°9}Ñ½Ñ…±}Ñ¥µ”œé9Q=Q0°ÑÉÕ•}…±Á¡„œé1A!}QIU°ÑÉÕ•}Ñ¡•Ñ„œéQ!Q}QIU°(€€€€€€€€€€É¡½}µ…¥¸œéI!=}5%8°ÑÉ…¥¹}Í…µÁ±•Ìœé9}QI%8°Ù…±}Í…µÁ±•Ìœé9}Y0°Ñ•ÍÑ}Í…µÁ±•Ìœé9}QMP°µ…¥¹}Í••‘Ìœé5%9}ML°(€€€€€€€€€€¹•…É}É¡½Ìœé9I}I!=L°±…ÍÑ}Á¡…Í”œé…ÉÌ¹Á¡…Í”°±…ÍÑ}ÉÕ¹Ñ¥µ•}ÌœéÑ¥µ”¹Ñ¥µ” ¤µĞÀ°Ñ½É œéÑ½É ¹}}Ù•ÉÍ¥½¹}|°‘•Ù¥”œéÍÑÈ¡Y%¥ô(€€€İ¥Ñ ½Á•¸¡=UP½˜‰•¹¡µ…É­}µ•Ñ…‘…Ñ…}í…ÉÌ¹Á¡…Í•ô¹©Í½¸œ°Üœ¤…Ì˜è©Í½¸¹‘ÕµÀ¡µ•Ñ„±˜±¥¹‘•¹ĞôÈ¤(€€€ÁÉ¥¹Ğ q¹=9œ±µ•Ñ„±™±ÕÍ õQÉÕ”¤(