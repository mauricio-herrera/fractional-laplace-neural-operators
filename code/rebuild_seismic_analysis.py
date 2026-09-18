#!/usr/bin/env python3
"""Rebuild the Chile seismic analysis for fLNO v2.1 from the Potin et al. relocated CSN catalogue.

Primary choices
---------------
* One homogeneous relocated catalogue for all four sequences.
* Official USGS mainshock anchors; the catalogue supplies aftershocks.
* 350-km radius, 365-day primary window, 180-day memory truncation.
* Sequence-specific completeness threshold = conservative maximum of:
  MAXC + 0.2, 90% goodness-of-fit threshold, and a b-value-stability threshold.
* Scalar daily Poisson Hawkes-Volterra fit with tempered Omori atom.
* Profile likelihood for branching ratio rho.
* Parametric bootstrap as a sensitivity analysis, not a nominal-coverage claim.
* Ordered along-strike zones from PCA + tertiles; no arbitrary k-means labels.
* Spatial ladder: M0 independent, M1 symmetric nearest-neighbor mixing,
  M2 directional north/south nearest-neighbor mixing. All coupling matrices are
  column stochastic and therefore have Perron root one.
"""
from __future__ import annotations
import json, math, hashlib, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/potin/CHILE_SEISMICITY_RELOCATED.csv"
OUT = ROOT / "results/seismic"
SEQOUT = ROOT / "data/chile_sequences_potin"
MS = ROOT / "ms"
OUT.mkdir(parents=True, exist_ok=True)
SEQOUT.mkdir(parents=True, exist_ok=True)

DAYS_PRIMARY = 365
RADIUS_KM = 350.0
L = 180
B_BOOT = int(os.environ.get("FLNO_BOOTSTRAP_B", "300"))
SEED = 20260918

ANCHORS = {
    "Tocopilla": dict(time="2007-11-14T15:40:50Z", lat=-22.247, lon=-69.890, depth=40.0, mag=7.7, event_id="usp000fshy"),
    "Maule": dict(time="2010-02-27T06:34:11Z", lat=-36.122, lon=-72.898, depth=22.9, mag=8.8, event_id="usp000h7rf"),
    "Iquique": dict(time="2014-04-01T23:46:47Z", lat=-19.610, lon=-70.769, depth=25.0, mag=8.2, event_id="usc000nzvd"),
    "Illapel": dict(time="2015-09-16T22:54:32Z", lat=-31.573, lon=-71.674, depth=22.4, mag=8.3, event_id="us20003k7a"),
}

BOUNDS_T = [(-10,5), (-6,1.5), (0.8,2.5), (math.log(.01),math.log(2.0)), (-12,math.log(2.0))]
BOUNDS_U = [(-10,5), (-6,1.5), (0.8,2.5), (math.log(.01),math.log(2.0))]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()

def haversine(lat1, lon1, lat2, lon2):
    R=6371.0
    p1=np.radians(lat1); p2=np.radians(lat2)
    dp=np.radians(np.asarray(lat2)-lat1); dl=np.radians(np.asarray(lon2)-lon1)
    aa=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(aa,0,1)))

def load_catalog():
    d=pd.read_csv(RAW)
    d['magnitude_type']=d['magnitude_type'].astype(str).str.strip()
    base=pd.to_datetime(dict(year=d.year, month=d.month, day=d.day, hour=d.hour, minute=d.minute), utc=True)
    d['time']=base+pd.to_timedelta(d['seconds'],unit='s')
    return d

def estimate_b(x, mc, bw=.1):
    x=np.asarray(x,float); x=x[np.isfinite(x)&(x>=mc-1e-9)]
    if len(x)<30: return np.nan
    den=x.mean()-(mc-bw/2)
    return np.log10(np.e)/den if den>0 else np.nan

def gft_score(x,mc,bw=.1):
    x=np.asarray(x,float); x=x[np.isfinite(x)&(x>=mc-1e-9)]
    if len(x)<50: return np.nan
    b=estimate_b(x,mc,bw)
    if not np.isfinite(b): return np.nan
    top=np.ceil(x.max()/bw)*bw
    th=np.arange(mc,top+1e-8,bw)
    obs=np.array([(x>=t-1e-9).sum() for t in th],float)
    pred=len(x)*10**(-b*(th-mc))
    return float(100*(1-np.sum(np.abs(obs-pred))/np.sum(obs)))

def completeness(x,bw=.1):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    lo=np.floor(x.min()/bw)*bw; hi=np.ceil(x.max()/bw)*bw
    edges=np.arange(lo,hi+bw*1.01,bw); hist,_=np.histogram(x,bins=edges)
    im=int(np.argmax(hist)); maxc=float(edges[im]+bw/2)
    cand=np.arange(max(2.0,maxc-.6),min(5.5,np.percentile(x,95)-.2)+1e-8,bw)
    scores=np.array([gft_score(x,m,bw) for m in cand])
    gft=next((float(m) for m,s in zip(cand,scores) if np.isfinite(s) and s>=90),np.nan)
    bs=np.array([estimate_b(x,m,bw) for m in cand])
    bstab=np.nan
    for i in range(max(0,len(cand)-5)):
        if i+5<len(cand) and np.isfinite(bs[i:i+6]).all() and abs(bs[i]-bs[i+1:i+6].mean())<=.05:
            bstab=float(cand[i]); break
    vals=[maxc+.2]+([gft] if np.isfinite(gft) else [])+([bstab] if np.isfinite(bstab) else [])
    chosen=float(np.ceil(max(vals)*10-1e-8)/10)
    return dict(maxc=maxc,gft90=gft,bstability=bstab,chosen=chosen)

def extract_catalog(d,name,days=DAYS_PRIMARY,radius=RADIUS_KM):
    a=ANCHORS[name]; t0=pd.Timestamp(a['time'])
    dist=haversine(a['lat'],a['lon'],d.latitude.values,d.longitude.values)
    tau=(d.time-t0).dt.total_seconds()/86400
    keep=(tau>60/86400)&(tau<=days)&(dist<=radius)&d.magnitude.notna()
    s=d.loc[keep].copy(); s['tau_days']=tau[keep]; s['dist_km']=dist[keep]
    return s

def daily_counts(s,mc,days=DAYS_PRIMARY):
    ss=s[s.magnitude>=mc-1e-9].copy()
    y,_=np.histogram(ss.tau_days,bins=np.arange(0,days+2))
    y=y.astype(float); y[0]+=1.0
    return ss,y

def kernel_weights(p,theta,c,L=L):
    ell=np.arange(1,L+1,dtype=float)
    m=(ell+c)**(-p)*np.exp(-theta*ell)
    return m,float(m.sum())

def poisson_nll(v,y,tempered=True):
    logmu,logrho,p,logc=v[:4]; theta=np.exp(v[4]) if tempered else 0.0
    mu=np.exp(logmu); rho=np.exp(logrho); c=np.exp(logc)
    m,G=kernel_weights(p,theta,c)
    z=np.convolve(y,m)[:len(y)]; H=np.zeros_like(y); H[1:]=z[:len(y)-1]
    lam=np.maximum(mu+(rho/G)*H,1e-12)
    return float(np.sum(lam[1:]-y[1:]*np.log(lam[1:])+gammaln(y[1:]+1)))

def fit_scalar(y,tempered=True,seed=0,warm=None,starts=10):
    rng=np.random.default_rng(seed); bounds=BOUNDS_T if tempered else BOUNDS_U
    mean=max(float(np.mean(y[1:])),1e-3); x0=[]
    if warm is not None:
        x0.append(np.clip(np.asarray(warm,float),[b[0] for b in bounds],[b[1] for b in bounds]))
    base=[np.log(mean*.2+1e-3),np.log(.7),1.3,np.log(.1)]+([np.log(.01)] if tempered else [])
    x0.append(np.asarray(base))
    for _ in range(starts):
        x=[np.log(mean*rng.uniform(.03,.5)+1e-5),np.log(rng.uniform(.3,1.15)),rng.uniform(.9,2.2),np.log(10**rng.uniform(-2,.3))]
        if tempered: x.append(np.log(10**rng.uniform(-4,np.log10(.4))))
        x0.append(np.asarray(x))
    best=None
    for x in x0:
        r=minimize(poisson_nll,x,args=(y,tempered),method='L-BFGS-B',bounds=bounds,
                   options={'maxiter':1800,'ftol':1e-11,'gtol':1e-7,'maxls':50})
        if best is None or r.fun<best.fun: best=r
    return best

def unpack_scalar(r,tempered=True):
    x=r.x
    return dict(mu=float(np.exp(x[0])),rho=float(np.exp(x[1])),p=float(x[2]),c=float(np.exp(x[3])),
                theta=float(np.exp(x[4])) if tempered else 0.0,nll=float(r.fun),x=x.copy())

def profile_rho(y,fit,tempered=True):
    xhat=fit.x.copy(); lr0=xhat[1]; bounds=BOUNDS_T if tempered else BOUNDS_U
    grid=np.linspace(max(bounds[1][0],lr0-1.05),min(bounds[1][1],lr0+.75),43)
    prof=[]; nuisance=np.delete(xhat,1)
    nb=[bounds[i] for i in range(len(bounds)) if i!=1]
    for lr in grid:
        def obj(q): return poisson_nll(np.insert(q,1,lr),y,tempered)
        rr=minimize(obj,nuisance,method='L-BFGS-B',bounds=nb,options={'maxiter':800,'ftol':1e-10})
        nuisance=rr.x; prof.append(rr.fun)
    prof=np.asarray(prof); dev=2*(prof-fit.fun); inside=np.exp(grid[dev<=3.841459])
    ci=(np.nan,np.nan) if len(inside)==0 else (float(inside.min()),float(inside.max()))
    return ci,np.exp(grid),dev

def simulate(y0,T,pars,rng):
    m,G=kernel_weights(pars['p'],pars['theta'],pars['c']); yy=np.zeros(T+1); yy[0]=y0
    for t in range(1,T+1):
        l=min(L,t); H=float(np.dot(m[:l],yy[t-1::-1][:l])); lam=max(pars['mu']+(pars['rho']/G)*H,1e-10)
        yy[t]=rng.poisson(lam)
    return yy

def refit_bootstrap(y, xwarm):
    return minimize(poisson_nll,np.asarray(xwarm,float),args=(y,True),method='L-BFGS-B',bounds=BOUNDS_T,
                    options={'maxiter':700,'ftol':1e-9,'gtol':1e-6,'maxls':30})

def bootstrap(y,fit,B,seed):
    pars=unpack_scalar(fit,True); rng=np.random.default_rng(seed); vals=[]; xw=fit.x
    for _ in range(B):
        yy=simulate(y[0],len(y)-1,pars,rng)
        rr=refit_bootstrap(yy,xw); vals.append(np.exp(rr.x[1]))
    vals=np.asarray(vals); bias=float(vals.mean()-pars['rho'])
    qlo,qhi=np.quantile(vals,[.025,.975]); basic=(float(2*pars['rho']-qhi),float(2*pars['rho']-qlo))
    return dict(B=B,bias=bias,rho_bc=float(pars['rho']-bias),basic_low=basic[0],basic_high=basic[1],
                sd=float(vals.std(ddof=1)),q025=float(qlo),q975=float(qhi)),vals

def local_xy(s,anchor):
    lat0=np.radians(anchor['lat'])
    x=(s.longitude.values-anchor['lon'])*111.32*np.cos(lat0)
    y=(s.latitude.values-anchor['lat'])*110.57
    return np.c_[x,y]

def along_strike_zones(s,name):
    X=local_xy(s,ANCHORS[name]); Xc=X-X.mean(0); cov=Xc.T@Xc/max(len(Xc)-1,1)
    ev,vec=np.linalg.eigh(cov); v=vec[:,np.argmax(ev)]
    if v[1]<0: v=-v
    score=Xc@v; q1,q2=np.quantile(score,[1/3,2/3]); z=np.digitize(score,[q1,q2])
    out=s.copy(); out['along_strike_km']=score; out['zone']=z+1
    return out,v,(q1,q2)

def C_matrix(mode,par):
    if mode=='M0': return np.eye(3),0.,0.
    if mode=='M1':
        e=1/(1+np.exp(-par)); en=es=e/2
    else:
        u,v=par; eu,ev=np.exp(np.clip([u,v],-20,20)); den=1+eu+ev; en,es=eu/den,ev/den
    C=np.zeros((3,3))
    for j in range(3):
        stay=1-en-es; C[j,j]+=stay
        if j<2: C[j+1,j]+=en
        else: C[j,j]+=en
        if j>0: C[j-1,j]+=es
        else: C[j,j]+=es
    return C,float(en),float(es)

def spatial_nll(v,Y,kpars,mode):
    q=3; mus=np.exp(v[:q]); rho=np.exp(v[q]); extra=v[q+1:]
    if mode=='M0': C,en,es=C_matrix('M0',0.)
    elif mode=='M1': C,en,es=C_matrix('M1',extra[0])
    else: C,en,es=C_matrix('M2',extra[:2])
    m,G=kernel_weights(kpars['p'],kpars['theta'],kpars['c']); T=len(Y); H=np.zeros_like(Y,dtype=float)
    for j in range(q):
        zz=np.convolve(Y[:,j],m)[:T]; H[1:,j]=zz[:T-1]
    lam=mus[None,:]+(rho/G)*(H@C.T); lam=np.maximum(lam[1:],1e-12); yy=Y[1:]
    return float(np.sum(lam-yy*np.log(lam)+gammaln(yy+1)))

def profile_eps_m1(Y,kpars,fit):
    eps_grid=np.r_[np.linspace(.001,.10,12),np.linspace(.12,.90,27)]
    vals=[]; warm=fit['x'][:4].copy(); bounds=[(-10,5)]*3+[(-6,1.5)]
    for e in eps_grid:
        logit=float(np.log(e/(1-e)))
        def obj(q):
            v=np.r_[q,logit]
            return spatial_nll(v,Y,kpars,'M1')
        rr=minimize(obj,warm,method='L-BFGS-B',bounds=bounds,options={'maxiter':700,'ftol':1e-10})
        warm=rr.x; vals.append(rr.fun)
    vals=np.asarray(vals); dev=2*(vals-fit['nll']); inside=eps_grid[dev<=3.841459]
    ci=(float(inside.min()),float(inside.max())) if len(inside) else (np.nan,np.nan)
    return ci,eps_grid,dev

def fit_spatial(Y,kpars,mode,seed=0):
    rng=np.random.default_rng(seed); q=3; means=np.maximum(Y[1:].mean(0)*.2,1e-3)
    n=4+(0 if mode=='M0' else 1 if mode=='M1' else 2)
    base=np.r_[np.log(means),np.log(kpars['rho'])]
    if mode=='M1': base=np.r_[base,-2.0]
    elif mode=='M2': base=np.r_[base,-2.5,-2.5]
    starts=[base]
    for _ in range(7):
        z=np.r_[np.log(means*rng.uniform(.5,2,3)),np.log(rng.uniform(.3,1.1))]
        if mode=='M1': z=np.r_[z,rng.uniform(-4,1)]
        elif mode=='M2': z=np.r_[z,rng.uniform(-4,1),rng.uniform(-4,1)]
        starts.append(z)
    bounds=[(-10,5)]*3+[(-6,1.5)]+([] if mode=='M0' else [(-8,4)] if mode=='M1' else [(-8,4),(-8,4)])
    best=None
    for z in starts:
        rr=minimize(spatial_nll,z,args=(Y,kpars,mode),method='L-BFGS-B',bounds=bounds,options={'maxiter':1500,'ftol':1e-11})
        if best is None or rr.fun<best.fun: best=rr
    C,en,es=C_matrix(mode,0 if mode=='M0' else best.x[4] if mode=='M1' else best.x[4:6])
    return dict(mode=mode,nll=float(best.fun),npar=n,AIC=float(2*n+2*best.fun),rho=float(np.exp(best.x[3])),
                north=en,south=es,C=C,x=best.x)

def make_zone_counts(zoned,days=DAYS_PRIMARY):
    Y=np.zeros((days+1,3),float); Y[0,:]=0
    for k in range(1,4):
        vals=zoned.loc[zoned.zone==k,'tau_days'].values
        Y[:,k-1],_=np.histogram(vals,bins=np.arange(0,days+2))
    Y[0,1]+=1
    return Y

def main():
    d=load_catalog()
    prov=dict(dataset_doi="10.5281/zenodo.13146436",article_doi="10.1785/0220240047",
              raw_sha256=sha256(RAW),events=int(len(d)),primary_days=DAYS_PRIMARY,
              radius_km=RADIUS_KM,memory_days=L,bootstrap_B=B_BOOT,anchors=ANCHORS)
    (OUT/'provenance.json').write_text(json.dumps(prov,indent=2))
    rows=[]; sens=[]; spatial=[]
    for ii,name in enumerate(ANCHORS):
        s0=extract_catalog(d,name); mcinfo=completeness(s0.magnitude.values); mc=mcinfo['chosen']
        s,y=daily_counts(s0,mc)
        zoned,axis,cuts=along_strike_zones(s,name); zoned.to_csv(SEQOUT/f'{name}_events.csv',index=False)
        qc=dict(sequence=name,n_raw_radius=len(s0),mc=mc,n_used=len(s),
                unique_xy=float(s[['latitude','longitude']].drop_duplicates().shape[0]/len(s)),
                lat_span=float(s.latitude.max()-s.latitude.min()),
                lon_span=float(s.longitude.max()-s.longitude.min()),
                depth_sd=float(s.depth.std()),maxc=mcinfo['maxc'],
                gft90=mcinfo['gft90'],bstability=mcinfo['bstability'])
        ft=fit_scalar(y,True,seed=SEED+ii); fu=fit_scalar(y,False,seed=SEED+100+ii)
        pt=unpack_scalar(ft,True); pu=unpack_scalar(fu,False)
        ci,grid,dev=profile_rho(y,ft,True); boot,bvals=bootstrap(y,ft,B_BOOT,SEED+1000*ii)
        daic=(2*5+2*ft.fun)-(2*4+2*fu.fun)
        row={**qc,**{k:pt[k] for k in ('mu','rho','p','c','theta','nll')},
             'profile_low':ci[0],'profile_high':ci[1],
             'deltaAIC_tempered_minus_untempered':float(daic),
             **{f'boot_{k}':v for k,v in boot.items()}}
        rows.append(row); np.savez(OUT/f'{name}_profile.npz',rho_grid=grid,deviance=dev)
        np.save(OUT/f'{name}_bootstrap_rho.npy',bvals)
        for T in (180,365,730):
            sT=extract_catalog(d,name,days=T)
            for dm in (0,.25,.5):
                mct=float(np.ceil((mc+dm)*10-1e-8)/10)
                _,yy=daily_counts(sT,mct,days=T)
                rr=fit_scalar(yy,True,seed=SEED+T+int(100*dm)+ii,starts=5)
                pp=unpack_scalar(rr,True)
                sens.append(dict(sequence=name,days=T,mc=mct,n=int(yy.sum()),rho=pp['rho'],p=pp['p'],theta=pp['theta']))
        Y=make_zone_counts(zoned)
        f0=fit_spatial(Y,pt,'M0',SEED+ii); f1=fit_spatial(Y,pt,'M1',SEED+20+ii); f2=fit_spatial(Y,pt,'M2',SEED+40+ii)
        ci_eps,eg,ed=profile_eps_m1(Y,pt,f1)
        spatial.append(dict(sequence=name,rho_M0=f0['rho'],rho_M1=f1['rho'],rho_M2=f2['rho'],
                            eps_M1=f1['north']+f1['south'],eps_low=ci_eps[0],eps_high=ci_eps[1],
                            north_M2=f2['north'],south_M2=f2['south'],
                            dAIC_M1_M0=f1['AIC']-f0['AIC'],dAIC_M2_M1=f2['AIC']-f1['AIC'],
                            axis_x=float(axis[0]),axis_y=float(axis[1]),
                            cut1=float(cuts[0]),cut2=float(cuts[1])))
        np.savez(OUT/f'{name}_spatial_matrices.npz',M0=f0['C'],M1=f1['C'],M2=f2['C'],eps_grid=eg,eps_deviance=ed)
    pd.DataFrame(rows).to_csv(OUT/'scalar_results.csv',index=False)
    pd.DataFrame(sens).to_csv(OUT/'sensitivity_results.csv',index=False)
    pd.DataFrame(spatial).to_csv(OUT/'spatial_results.csv',index=False)
    print(pd.DataFrame(rows)[['sequence','mc','n_used','rho','profile_low','profile_high',
                              'boot_rho_bc','boot_basic_low','boot_basic_high','p','theta',
                              'deltaAIC_tempered_minus_untempered']].to_string(index=False))
    print('\nSpatial ladder')
    print(pd.DataFrame(spatial).to_string(index=False))

if __name__=='__main__':
    main()
