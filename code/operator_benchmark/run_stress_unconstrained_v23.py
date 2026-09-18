from pathlib import Path
import importlib.util
import numpy as np, pandas as pd, torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bench', HERE/'run_operator_benchmark.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
OUT=HERE.parent.parent/'results'/'operator_benchmark'; OUT.mkdir(parents=True,exist_ok=True)
b.N_TRAIN=12; b.N_VAL=12; b.N_TEST=32
rho=0.97; nseeds=20; noise_rel=0.10
rows=[]
for s in range(nseeds):
    seed=1000+s; b.seed_all(seed)
    U,Y=b.generate_dataset(rho, 22000+seed)
    rng=np.random.default_rng(71000+seed)
    sigma=noise_rel*np.std(Y[:b.N_TRAIN,:b.NTRAIN_T])
    Yn=Y.copy(); Yn[:b.N_TRAIN,:b.NTRAIN_T]+=rng.normal(0,sigma,Yn[:b.N_TRAIN,:b.NTRAIN_T].shape).astype(np.float32)
    model=b.UnconstrainedRational(12).to(b.DEVICE)
    with torch.no_grad():
        rates,_=model.pars()
        q0=torch.tensor(rng.normal(0.0,0.20,size=12),dtype=torch.float32,device=b.DEVICE)*rates
        model.q.copy_(q0)
    Ut=torch.tensor(U,device=b.DEVICE); Yt=torch.tensor(Yn,device=b.DEVICE)
    train_idx=np.arange(b.N_TRAIN); val_idx=np.arange(b.N_TRAIN,b.N_TRAIN+b.N_VAL)
    opt=torch.optim.Adam(model.parameters(),lr=0.012); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=140,eta_min=0.0006)
    best=None; bestloss=np.inf; rtrain=np.random.default_rng(seed+999)
    for ep in range(140):
        perm=rtrain.permutation(train_idx); model.train()
        for st in range(0,len(perm),12):
            idx=torch.tensor(perm[st:st+12],device=b.DEVICE)
            pred=model(Ut[idx,:b.NTRAIN_T],b.NTRAIN_T); loss=torch.nn.functional.mse_loss(pred,Yt[idx,:b.NTRAIN_T])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        sched.step()
        if ep%5==0 or ep==139:
            model.eval()
            with torch.no_grad():
                vi=torch.tensor(val_idx,device=b.DEVICE); vl=torch.nn.functional.mse_loss(model(Ut[vi,:b.NTRAIN_T],b.NTRAIN_T),Yt[vi,:b.NTRAIN_T]).item()
            if np.isfinite(vl) and vl<bestloss:
                bestloss=vl; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best is not None: model.load_state_dict(best)
    def pf(u_np,nt):
        with torch.no_grad(): return model(torch.tensor(u_np,device=b.DEVICE),nt).cpu().numpy()
    row,_,_=b.eval_model('UnconstrainedRational',model,pf,U,Y,rho,0.0)
    root=float(row['max_root_real']); unstable=root>=0
    rows.append({'seed':seed,'n_train':b.N_TRAIN,'noise_rel':noise_rel,'max_root_real':root,'unstable':int(unstable),'tail_relL2':row['tail_relL2'],'in_relL2':row['relL2_in'],'dc_mass_over_c':row['dc_mass_over_c'],'branch_error':row['branch_margin_abs_error'],'val_loss':bestloss})
    print(seed,'root',root,'unstable',int(unstable),'tail',row['tail_relL2'],flush=True)
df=pd.DataFrame(rows); df.to_csv(OUT/'stress_unconstrained_v23.csv',index=False)
s={'runs':len(df),'unstable_runs':int(df.unstable.sum()),'unstable_fraction':float(df.unstable.mean()),'max_root_real_max':float(df.max_root_real.max()),'max_root_real_median':float(df.max_root_real.median()),'tail_median':float(df.tail_relL2.median()),'tail_q90':float(df.tail_relL2.quantile(.9))}
pd.DataFrame([s]).to_csv(OUT/'stress_unconstrained_v23_summary.csv',index=False)
print(s)
