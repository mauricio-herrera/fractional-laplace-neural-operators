"""
fLNO epidemic application, scalar stage: Chile, national daily cases (OWID/JHU).

RENEWAL MODEL (the epidemiological reading of the Volterra intensity layer)
---------------------------------------------------------------------------
   I_t ~ NegBin( lambda_t, kappa ),
   lambda_t = delta_{dow(t)} * R_{k(t)} * Lambda_t,
   Lambda_t = sum_{l=1}^{Lg} m_l I_{t-l},   sum_l m_l = 1,
so the window gains R_k ARE the branching diagnostics rho of the fLNO layer,
i.e. the effective reproduction number of window k. The kernel m is LEARNED
jointly instead of frozen from the generation-interval literature.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist
import os

plt.rcParams.update({'font.family':'serif','font.size':10,'axes.titleweight':'bold',
                     'figure.dpi':120,'savefig.dpi':300,'savefig.bbox':'tight'})
NAVY, ORANGE, GRAY = '#1A2F4A', '#E65100', '#546E7A'
HERE=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
OUT=os.path.join(os.path.dirname(HERE),"results")
COUNTRY="Chile"; T0,T1="2020-03-15","2022-03-01"; Lg=21; WIN=28

df=pd.read_csv(f"{HERE}/new_cases.csv",parse_dates=["date"]).set_index("date")
y=df[COUNTRY].loc[T0:T1].astype(float).clip(lower=0).fillna(0.0)
dates=y.index; I=y.to_numpy(); T=len(I); dow=np.array([d.weekday() for d in dates])
K=(T-Lg)//WIN
def win_of(t): return min((t-Lg)//WIN,K-1)
widx=np.array([win_of(t) for t in range(T)])

def kernel(b,d):
    l=np.arange(1,Lg+1,dtype=float)
    w=np.exp(-b*l)-np.exp(-(b+d)*l); w=np.maximum(w,1e-300)
    return w/w.sum()
def conv_history(I,m):
    z=np.convolve(I,m); Lam=np.zeros(len(I)); Lam[1:]=z[:len(I)-1]; return Lam
def negbin_nll(yv,lam,kappa):
    lam=np.maximum(lam,1e-8)
    return -np.sum(gammaln(yv+kappa)-gammaln(kappa)-gammaln(yv+1)
                   +kappa*np.log(kappa/(kappa+lam))+yv*np.log(lam/(kappa+lam)))
def unpack(p):
    R=np.exp(p[:K]); b,d=np.exp(p[K]),np.exp(p[K+1]); kappa=np.exp(p[K+2])
    dow_free=p[K+3:K+9]; delta=np.exp(np.r_[dow_free,-dow_free.sum()])
    return R,b,d,kappa,delta
def nll(p):
    R,b,d,kappa,delta=unpack(p); Lam=conv_history(I,kernel(b,d))
    t=np.arange(Lg,T); lam=delta[dow[t]]*R[widx[t]]*Lam[t]
    return negbin_nll(I[t],lam,kappa)

gm_mean,gm_sd=5.2,1.9
shape,scale=(gm_mean/gm_sd)**2,gm_sd**2/gm_mean
w0=np.diff(gamma_dist.cdf(np.arange(0,Lg+1),a=shape,scale=scale)); w0/=w0.sum()
Lam0=conv_history(I,w0); cori=[]
for k in range(K):
    t=np.arange(Lg,T)[widx[np.arange(Lg,T)]==k]
    a_post=1.0+I[t].sum(); b_post=0.2+Lam0[t].sum()
    cori.append((a_post/b_post,gamma_dist.ppf(.025,a=a_post,scale=1/b_post),
                 gamma_dist.ppf(.975,a=a_post,scale=1/b_post)))
cori=np.array(cori)

p0=np.r_[np.log(np.maximum(cori[:,0],.05)),np.log(.20),np.log(.35),np.log(10.0),np.zeros(6)]
best=None
for scale_k in (1.0,.5):
    p_init=p0.copy(); p_init[K]+=np.log(scale_k)
    rr=minimize(nll,p_init,method="L-BFGS-B",options=dict(maxiter=800,maxfun=200000))
    if best is None or rr.fun<best.fun: best=rr
R_hat,b_hat,d_hat,kap_hat,delta_hat=unpack(best.x)
m_hat=kernel(b_hat,d_hat); lags=np.arange(1,Lg+1)
gi_mean=float(np.sum(lags*m_hat)); gi_sd=float(np.sqrt(np.sum((lags-gi_mean)**2*m_hat)))

def profile_Rk(k):
    t=np.arange(Lg,T)[widx[np.arange(Lg,T)]==k]; Lam=conv_history(I,m_hat)
    base=delta_hat[dow[t]]*Lam[t]
    def nll_k(R): return negbin_nll(I[t],R*base,kap_hat)
    Rk=R_hat[k]; grid=Rk*np.exp(np.linspace(-.5,.5,41))
    prof=np.array([nll_k(r) for r in grid]); dev=2*(prof-prof.min()); inside=grid[dev<=3.841]
    if inside.min()==grid.min() or inside.max()==grid.max():
        grid=Rk*np.exp(np.linspace(-1.2,1.2,81)); prof=np.array([nll_k(r) for r in grid])
        dev=2*(prof-prof.min()); inside=grid[dev<=3.841]
    return float(inside.min()),float(inside.max())

CIs=np.array([profile_Rk(k) for k in range(K)])
res=pd.DataFrame({"window_start":[dates[Lg+k*WIN].date() for k in range(K)],
                  "R_flno":R_hat,"ci_lo":CIs[:,0],"ci_hi":CIs[:,1],
                  "R_cori":cori[:,0],"cori_lo":cori[:,1],"cori_hi":cori[:,2]})
res.to_csv(f"{OUT}/chile_Rt_windows.csv",index=False)

fig,ax=plt.subplots(2,1,figsize=(11.5,7.2),height_ratios=[2.2,1],sharex=False)
tt=[dates[Lg+k*WIN+WIN//2] for k in range(K)]
ax[0].plot(dates,I/1000,color=GRAY,lw=.8,alpha=.6); ax[0].set_ylabel('new cases (thousands)',color=GRAY)
ax2=ax[0].twinx(); ax2.axhline(1.0,color='k',ls='--',lw=1)
for k in range(K): ax2.plot([tt[k],tt[k]],CIs[k],color=NAVY,lw=2.4,alpha=.8)
ax2.plot(tt,R_hat,'o-',color=NAVY,ms=4,lw=1.2,label='fLNO joint fit')
ax2.plot(tt,cori[:,0],'s--',color=ORANGE,ms=3.5,lw=1,label='Cori fixed-kernel baseline')
ax2.set_ylabel(r'$R_t$'); ax2.set_ylim(0,2.6); ax2.legend(fontsize=8,loc='upper right')
ax[1].bar(lags-.18,m_hat,.36,color=NAVY,label=f'learned kernel (mean {gi_mean:.1f} d, sd {gi_sd:.1f} d)')
ax[1].bar(lags+.18,w0,.36,color=ORANGE,alpha=.8,label='literature gamma')
ax[1].set_xlabel('lag (days)'); ax[1].set_ylabel('weight'); ax[1].legend(fontsize=8)
plt.tight_layout(); os.makedirs(f"{OUT}/figures",exist_ok=True); plt.savefig(f"{OUT}/figures/fig_epi_chile_scalar.png")
print(f"NLL={best.fun:.3f}; kernel mean={gi_mean:.3f}; sd={gi_sd:.3f}; windows excluding R=1={np.sum((CIs[:,0]>1)|(CIs[:,1]<1))}/{K}")
