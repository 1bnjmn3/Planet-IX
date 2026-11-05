"""Audit check for ORIGINAL_PROJECT_AUDIT.md. Run from anywhere: python rewrite-analysis/audit/audit_warp.py"""
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
ORIG = str(_ROOT / "Benjamins code") + "/"
HELPERS = str(_ROOT / "Planet_IX" / "planetx_helpers.py")
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
D=ORIG
df=pd.read_csv(D+'live_mpc_data.csv')
w=df[(df.a>250)&(df.q>40)&(df.i>14)&(df.i<22)]
print('N warp',len(w)); print(w[['ID','a','i','Node']].to_string(index=False))
def model(k1,k2,pa=500.0):
    def f(a,inc,mE):
        tg=1/a**3.5*k1; tp=mE*3e-6*a**2/pa**3*k2; return tp*inc/(tg+tp)
    return f
for k1,k2 in [(1e8,1e10),(1e8,1e9),(1e8,1e11),(1e9,1e10)]:
    p,c=curve_fit(model(k1,k2),w.a,w.i,p0=[20,5],bounds=([0,.1],[90,20]))
    print(f'k_giants={k1:.0e} k_planet={k2:.0e}: inc={p[0]:.2f} mass={p[1]:.3f} ME  sd={np.sqrt(np.diag(c))}')
# also with mass bounds removed
p,c=curve_fit(model(1e8,1e10),w.a,w.i,p0=[20,5],bounds=([0,1e-6],[90,1e6]))
print('unbounded mass fit',p, np.sqrt(np.diag(c)))
# chi2 of best vs flat line at mean
f=model(1e8,1e10); p,c=curve_fit(f,w.a,w.i,p0=[20,5],bounds=([0,.1],[90,20]))
print('rss model',((w.i-f(w.a,*p))**2).sum(),' rss constant mean',((w.i-w.i.mean())**2).sum(), 'mean i',w.i.mean(),'std',w.i.std())
# ratio torque_p/torque_g at a=250..600
for a in [250,400,600]: print(a,'ratio tp/tg', (p[1]*3e-6*a**2/500**3*1e10)/(1/a**3.5*1e8))
# physical ratio: omega_p/omega_g = m_p a^5/(a_p^3 * sum m_j a_j^2)  (Earth masses, AU)
J=317.8*5.2**2+95.2*9.5**2+14.5*19.2**2+17.1*30.1**2
for a in [250,400,600]:
    code=(3e-6*a**2/500**3*1e10)/(1/a**3.5*1e8); phys=a**5/(500**3*J)
    print(a,'per-Earth-mass ratio code',code,'physical (Laplace-Lagrange, e_p=0)',phys,'overstate x',code/phys)
# fit with physical torque ratio
def fphys(a,inc,mE):
    r=mE*a**5/(500**3*J); return r*inc/(1+r)
p2,c2=curve_fit(fphys,w.a,w.i,p0=[20,5],bounds=([0,.01],[90,1e4]))
print('physical-ratio fit inc,mass',p2,np.sqrt(np.diag(c2)))
# Null test for circularity: inclinations uniform in (14,22), independent of a
rng=np.random.default_rng(1); ms=[];incs=[]
for k in range(2000):
    fi=rng.uniform(14,22,len(w))
    try:
        pp,_=curve_fit(f,w.a.values,fi,p0=[20,5],bounds=([0,.1],[90,20])); ms.append(pp[1]); incs.append(pp[0])
    except Exception: pass
ms=np.array(ms);incs=np.array(incs)
print('NULL (i ~ U(14,22), no planet): fitted inc median',np.median(incs),'16-84',np.percentile(incs,[16,84]),' fitted mass median',np.median(ms),'16-84',np.percentile(ms,[16,84]), 'frac within 0.5-5 ME',((ms>.5)&(ms<5)).mean(), 'frac inc in 14-22',((incs>14)&(incs<22)).mean())
