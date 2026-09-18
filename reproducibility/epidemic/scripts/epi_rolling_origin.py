"""Executed rolling-origin evaluation for the Italian network models M0 and M2.

Protocol
--------
For origin k=10,...,17 (1-based count of fitted 28-day windows), fit all
shared parameters and R_1,...,R_k on windows 1..k, then score window k+1 one
step ahead using R-persistence (R_{k+1|k}=R_k). The score is the NB log score
per region-day observation. M2 is Perron-normalized exactly by the spectral
radius at every optimization step. Optimization is deterministic, float64,
Adam warm-up followed by L-BFGS, with the same Cori-style initialization rule
at every origin. No future outcome data enter a fit.
"""
from pathlib import Path
import json, time
import numpy as np, pandas as pd
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist
import torch

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'; RESULTS=ROOT/'results'
T0,T1='2020-09-01','2022-03-01'; Lg,WIN=21,28
popdf=pd.read_csv(DATA/'italy_population_2020.csv')
POP=dict(zip(popdf.region,popdf.population_millions))
raw=pd.read_csv(DATA/'dpc-covid19-ita-regioni.csv',parse_dates=['data']); raw['date']=raw['data'].dt.normalize()
piv=raw.pivot_table(index='date',columns='denominazione_regione',values='nuovi_positivi',aggfunc='sum').loc[T0:T1]
regions=[r for r in piv.columns if r in POP]; piv=piv[regions].clip(lower=0).fillna(0.0)
I=piv.to_numpy(float); dates=piv.index; T,q=I.shape; dow=np.array([d.weekday() for d in dates])
K=(T-Lg)//WIN; widx=np.minimum((np.arange(T)-Lg)//WIN,K-1)
p_share=np.array([POP[r] for r in regions]); p_share/=p_share.sum()
lat=raw.groupby('denominazione_regione')['lat'].first()[regions].to_numpy(); lon=raw.groupby('denominazione_regione')['long'].first()[regions].to_numpy()
R_earth=6371.0
def hav(la1,lo1,la2,lo2):
    la1,lo1,la2,lo2=map(np.radians,(la1,lo1,la2,lo2)); return 2*R_earth*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
D=hav(lat[:,None],lon[:,None],lat[None,:],lon[None,:])

def kernel_np(b,d):
    l=np.arange(1,Lg+1,dtype=float); w=np.exp(-b*l)-np.exp(-(b+d)*l); w=np.maximum(w,1e-300); return w/w.sum()
def conv_np(X,m):
    out=np.zeros_like(X,dtype=float)
    for lag in range(1,Lg+1): out[lag:]+=m[lag-1]*X[:-lag]
    return out
def A_np(eps,ell):
    W=p_share[None,:]*np.exp(-D/ell); np.fill_diagonal(W,0.0); W=W/W.sum(1,keepdims=True)
    A=(1-eps)*np.eye(q)+eps*(p_share[:,None]*W)*q
    rho=float(np.max(np.abs(np.linalg.eigvals(A))))
    return A/rho
def nb_nll_np(y,lam,kap):
    lam=np.maximum(lam,1e-8)
    return -np.sum(gammaln(y+kap)-gammaln(kap)-gammaln(y+1)+kap*np.log(kap/(kap+lam))+y*np.log(lam/(kap+lam)))

torch.set_default_dtype(torch.float64); torch.manual_seed(0); torch.set_num_threads(1)
It=torch.tensor(I); Dt=torch.tensor(D); pst=torch.tensor(p_share); dowt=torch.tensor(dow,dtype=torch.long); eye=torch.eye(q)
def tkernel(logb,logd):
    l=torch.arange(1,Lg+1); b=torch.exp(logb); d=torch.exp(logd); w=torch.exp(-b*l)-torch.exp(-(b+d)*l); return w/w.sum()
def tconv(X,m):
    out=torch.zeros_like(X)
    for lag in range(1,Lg+1): out[lag:]+=m[lag-1]*X[:-lag]
    return out
def tA(logit_eps,logell):
    eps=torch.sigmoid(logit_eps); ell=torch.exp(logell)
    W=pst[None,:]*torch.exp(-Dt/ell); W=W*(1-eye); W=W/W.sum(1,keepdim=True)
    A=(1-eps)*eye+eps*(pst[:,None]*W)*q
    v=torch.ones(q,dtype=torch.float64)/np.sqrt(q)
    for _ in range(30):
        v2=A@v; v=v2/(torch.linalg.norm(v2)+1e-15)
    rho=(v@(A@v))/(v@v)
    return A/rho,eps,ell
def nb_nll_t(y,lam,kap):
    lam=torch.clamp(lam,min=1e-8)
    return -(torch.lgamma(y+kap)-torch.lgamma(kap)-torch.lgamma(y+1)+kap*torch.log(kap/(kap+lam))+y*torch.log(lam/(kap+lam))).sum()

gm=gamma_dist(a=(5.2/1.9)**2,scale=1.9**2/5.2); w0=np.diff(gm.cdf(np.arange(0,Lg+1))); w0/=w0.sum()

def fit(mode,Ktr):
    Ttr=Lg+Ktr*WIN; X=It[:Ttr]; tt=np.arange(Lg,Ttr); wtr=np.minimum((np.arange(Ttr)-Lg)//WIN,Ktr-1)
    wt=torch.tensor(wtr[tt],dtype=torch.long); ti=torch.tensor(tt,dtype=torch.long)
    In=I[:Ttr].sum(1); Lam0=np.r_[0,np.convolve(In,w0)[:Ttr-1]]
    Rc=np.array([(1+In[tt[wtr[tt]==k]].sum())/(.2+Lam0[tt[wtr[tt]==k]].sum()) for k in range(Ktr)])
    pars=[torch.nn.Parameter(torch.tensor(np.log(np.maximum(Rc,.05)))),torch.nn.Parameter(torch.tensor(np.log(.25))),torch.nn.Parameter(torch.tensor(np.log(.4))),torch.nn.Parameter(torch.tensor(np.log(8.))),torch.nn.Parameter(torch.zeros(6))]
    if mode=='M2': pars += [torch.nn.Parameter(torch.tensor(0.0)),torch.nn.Parameter(torch.tensor(np.log(150.)))]
    def loss():
        R=torch.exp(pars[0]); m=tkernel(pars[1],pars[2]); kap=torch.exp(pars[3]); df=pars[4]; delta=torch.exp(torch.cat([df,-df.sum().view(1)])); Lam=tconv(X,m)
        if mode=='M2': A,_,_=tA(pars[5],pars[6]); Lam=Lam@A.T
        lam=delta[dowt[ti],None]*R[wt,None]*Lam[ti]
        return nb_nll_t(X[ti],lam,kap)
    opt=torch.optim.Adam(pars,lr=.03)
    for _ in range(150): opt.zero_grad(); L=loss(); L.backward(); opt.step()
    opt2=torch.optim.LBFGS(pars,lr=.5,max_iter=30,line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-9)
    def closure(): opt2.zero_grad(); L=loss(); L.backward(); return L
    opt2.step(closure); final=float(loss().detach())
    with torch.no_grad():
        R=torch.exp(pars[0]).numpy(); b=float(torch.exp(pars[1])); d=float(torch.exp(pars[2])); kap=float(torch.exp(pars[3])); df=pars[4]; delta=torch.exp(torch.cat([df,-df.sum().view(1)])).numpy(); eps=ell=None
        if mode=='M2': eps=float(torch.sigmoid(pars[5])); ell=float(torch.exp(pars[6]))
    return (R,b,d,kap,delta,eps,ell),final

def score(mode,fit,Ktr):
    R,b,d,kap,delta,eps,ell=fit; Lam=conv_np(I,kernel_np(b,d))
    if mode=='M2': Lam=Lam@A_np(eps,ell).T
    gt=np.arange(Lg,T); tk=gt[widx[gt]==Ktr]
    lam=delta[dow[tk],None]*R[-1]*Lam[tk]
    ll=-nb_nll_np(I[tk],lam,kap)
    return float(ll/I[tk].size),len(tk)*q,str(dates[tk[0]].date()),str(dates[tk[-1]].date())

rows=[]; t0=time.time(); partial=RESULTS/'italy_rolling_origin_partial.csv'; completed=set()
if partial.exists():
    old=pd.read_csv(partial); rows=old.to_dict('records'); completed=set(int(x) for x in old.origin_train_windows)
for k in range(10,18):
    if k in completed:
        print(f'k={k} already completed; skipping', flush=True); continue
    row={'origin_train_windows':k}
    for mode in ('M0','M2'):
        f,nll=fit(mode,k); sc,nobs,d0,d1=score(mode,f,k)
        row[f'{mode}_logscore']=sc; row[f'{mode}_nll_train']=nll
        if mode=='M2': row['eps_hat']=f[5]; row['ell_hat_km']=f[6]
        row['score_nobs']=nobs; row['score_start']=d0; row['score_end']=d1
    row['gain_M2_minus_M0']=row['M2_logscore']-row['M0_logscore']; rows.append(row)
    pd.DataFrame(rows).sort_values('origin_train_windows').to_csv(partial,index=False)
    print(f"k={k} {d0}..{d1}: gain={row['gain_M2_minus_M0']:+.5f}",flush=True)

df=pd.DataFrame(rows).sort_values('origin_train_windows'); df.to_csv(RESULTS/'italy_rolling_origin.csv',index=False)
summary={'origins':list(range(10,18)),'mean_gain_M2_minus_M0':float(df.gain_M2_minus_M0.mean()),'median_gain_M2_minus_M0':float(df.gain_M2_minus_M0.median()),'positive_origins':int((df.gain_M2_minus_M0>0).sum()),'n_origins':8,'mean_M0_logscore':float(df.M0_logscore.mean()),'mean_M2_logscore':float(df.M2_logscore.mean()),'elapsed_seconds':float(time.time()-t0),'optimizer':'torch float64: Adam(150) + L-BFGS(max_iter=30)','protocol':'train windows 1..k, score k+1, k=10..17; R-persistence; NB log score per region-day'}
(RESULTS/'italy_rolling_origin_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
