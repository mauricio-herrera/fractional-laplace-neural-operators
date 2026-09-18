import argparse, json, pandas as pd
import run_operator_benchmark as b
ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--rho',type=float,default=b.RHO_MAIN); ap.add_argument('--epochs',type=int,default=None); ap.add_argument('--tag',default='single'); ap.add_argument('--data-seed',type=int,default=None)
a=ap.parse_args()
U,Y=b.generate_dataset(a.rho, a.data_seed if a.data_seed is not None else 1000+a.seed)
m,pf,tt,_=b.train_model(a.model,a.seed,U,Y,epochs=a.epochs)
row,_,_=b.eval_model(a.model,m,pf,U,Y,a.rho,tt)
row.update({'seed':a.seed,'phase':a.tag})
path=b.OUT/f'{a.tag}_{a.model}_seed{a.seed}_rho{a.rho:.2f}.csv'
pd.DataFrame([row]).to_csv(path,index=False)
print(json.dumps(row,indent=2,default=float))
