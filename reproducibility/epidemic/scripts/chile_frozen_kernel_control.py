"""Frozen-kernel negative-binomial control for the Chile scalar renewal model.
Uses the same 24 windows, NB likelihood, day-of-week constraint and dispersion
parameter as the learned-kernel fit, but fixes the generation kernel to a
Gamma(mean=5.2 d, sd=1.9 d). Writes a reproducible AIC comparison.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'; RESULTS=ROOT/'results'
COUNTRY='Chile'; T0,T1='2020-03-15','2022-03-01'; Lg=21; WIN=28

df=pd.read_csv(DATA/'new_cases.csv',parse_dates=['date']).set_index('date')
y=df[COUNTRY].loc[T0:T1].astype(float).clip(lower=0).fillna(0.0)
dates=y.index; I=y.to_numpy(); T=len(I); dow=np.array([d.weekday() for d in dates])
K=(T-Lg)//WIN
widx=np.array([min((t-Lg)//WIN,K-1) for t in range(T)])

def conv_history(x,m):
    z=np.convolve(x,m); out=np.zeros(len(x)); out[1:]=z[:len(x)-1]; return out

def negbin_nll(yv,lam,kappa):
    lam=np.maximum(lam,1e-8)
    return -np.sum(gammaln(yv+kappa)-gammaln(kappa)-gammaln(yv+1)
                   +kappa*np.log(kappa/(kappa+lam))+yv*np.log(lam/(kappa+lam)))

gm_mean,gm_sd=5.2,1.9
shape,scale=(gm_mean/gm_sd)**2,gm_sd**2/gm_mean
w0=np.diff(gamma_dist.cdf(np.arange(0,Lg+1),a=shape,scale=scale)); w0=w0/w0.sum()
Lam=conv_history(I,w0)
t=np.arange(Lg,T)
cori=[]
for k in range(K):
    tk=t[widx[t]==k]
    a=1.0+I[tk].sum(); b=0.2+Lam[tk].sum(); cori.append(a/b)
cori=np.array(cori)

def unpack(p):
    R=np.exp(p[:K]); kappa=np.exp(p[K]); df=p[K+1:K+7]
    delta=np.exp(np.r_[df,-df.sum()])
    return R,kappa,delta

def nll(p):
    R,kappa,delta=unpack(p)
    lam=delta[dow[t]]*R[widx[t]]*Lam[t]
    return negbin_nll(I[t],lam,kappa)

p0=np.r_[np.log(np.maximum(cori,.05)),np.log(10.0),np.zeros(6)]
r=minimize(nll,p0,method='L-BFGS-B',options=dict(maxiter=1000,maxfun=300000))
R,kappa,delta=unpack(r.x)
k_frozen=K+1+6
AIC_frozen=2*k_frozen+2*r.fun

def learned_kernel(b,d):
    l=np.arange(1,Lg+1,dtype=float); w=np.exp(-b*l)-np.exp(-(b+d)*l); w=np.maximum(w,1e-300); return w/w.sum()
def unpackL(p):
    R=np.exp(p[:K]); b,d=np.exp(p[K]),np.exp(p[K+1]); kap=np.exp(p[K+2]); df=p[K+3:K+9]
    delta=np.exp(np.r_[df,-df.sum()]); return R,b,d,kap,delta
def nllL(p):
    R,b,d,kap,delta=unpackL(p); L=conv_history(I,learned_kernel(b,d))
    return negbin_nll(I[t],delta[dow[t]]*R[widx[t]]*L[t],kap)
pL=np.r_[np.log(np.maximum(cori,.05)),np.log(.20),np.log(.35),np.log(10.0),np.zeros(6)]
best=None
for sk in (1.0,.5):
    q=pL.copy(); q[K]+=np.log(sk)
    rr=minimize(nllL,q,method='L-BFGS-B',options=dict(maxiter=1000,maxfun=300000))
    if best is None or rr.fun<best.fun: best=rr
RL,bL,dL,kL,delL=unpackL(best.x)
k_learned=K+2+1+6
AIC_learned=2*k_learned+2*best.fun
m=learned_kernel(bL,dL); l=np.arange(1,Lg+1,dtype=float)
mean=float((l*m).sum()); sd=float(np.sqrt(((l-mean)**2*m).sum()))
out={
  'nll_learned':float(best.fun),'npar_learned':int(k_learned),'AIC_learned':float(AIC_learned),
  'nll_frozen':float(r.fun),'npar_frozen':int(k_frozen),'AIC_frozen':float(AIC_frozen),
  'delta_AIC_learned_minus_frozen':float(AIC_learned-AIC_frozen),
  'learned_kernel_mean_days':mean,'learned_kernel_sd_days':sd,
  'kappa_learned':float(kL),'kappa_frozen':float(kappa),
  'optimizer_success_learned':bool(best.success),'optimizer_success_frozen':bool(r.success)
}
(RESULTS/'chile_kernel_aic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
