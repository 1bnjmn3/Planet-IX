"""Audit check for ORIGINAL_PROJECT_AUDIT.md. Run from anywhere: python rewrite-analysis/audit/audit_misc.py"""
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
ORIG = str(_ROOT / "Benjamins code") + "/"
HELPERS = str(_ROOT / "Planet_IX" / "planetx_helpers.py")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
# 30 galactic plane: code's track vs true galactic plane (l=0..360,b=0) in RA/Dec
def code_track(inc,node):
    inc,node=np.radians(inc),np.radians(node); v=np.linspace(0,2*np.pi,2000)
    x=np.cos(node)*np.cos(v)-np.sin(node)*np.sin(v)*np.cos(inc); y=np.sin(node)*np.cos(v)+np.cos(node)*np.sin(v)*np.cos(inc); z=np.sin(v)*np.sin(inc)
    e=np.radians(23.439); ye=y*np.cos(e)-z*np.sin(e); ze=y*np.sin(e)+z*np.cos(e)
    return np.c_[x,ye,ze]
# true galactic plane: rotate from galactic to ICRS
Rg=np.array([[-0.0548755604,0.4941094279,-0.8676661490],[-0.8734370902,-0.4448296300,-0.1980763734],[-0.4838350155,0.7469822445,0.4559837762]])  # equatorial->galactic
l=np.radians(np.linspace(0,360,2000)); g=np.c_[np.cos(l),np.sin(l),0*l]; eq=g@Rg  # galactic->equatorial
ct=code_track(62.87,282.85)
d=np.degrees(np.arccos(np.clip(ct@eq.T,-1,1))).min(1)
print('30 galactic track: max angular error vs true galactic plane %.1f deg, median %.1f'%(d.max(),np.median(d)))
gc=np.array([np.cos(np.radians(-28.94))*np.cos(np.radians(266.40)),np.cos(np.radians(-28.94))*np.sin(np.radians(266.40)),np.sin(np.radians(-28.94))])
print('   distance of Galactic Centre from code track %.1f deg'%np.degrees(np.arccos(np.clip(ct@gc,-1,1))).min())
# 19/22 secular timescale
P=300**1.5; n=2*np.pi/P; w=0.75*n*(5*3.003e-6)*(300/500)**3/(1-0.25**2)**1.5
print('19: P9-driven nodal precession period at a=300AU: %.2f Gyr; 500 kyr = %.3f%% of it'%(2*np.pi/w/1e9, 5e5/(2*np.pi/w)*100))
print('22: rebound default G=1 so t=250000 code units = %.0f yr'%(250000/(2*np.pi)))
# 09 classifier fed pure random data
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv(ORIG+'processed_etnos.csv'); r=df[(df.a>230)&(df.q>30)]
rng=np.random.default_rng(0)
def world(n,clu):
    v=rng.vonmises(np.radians(60),2.0,n) if clu else rng.uniform(0,2*np.pi,n)
    return pd.DataFrame({'a':rng.choice(r.a,n),'e':rng.choice(r.e,n),'i_rad':rng.choice(r.i_rad,n),'varpi_sin':np.sin(v),'varpi_cos':np.cos(v),'label':int(clu)})
tr=pd.concat([world(10000,1),world(10000,0)]).sample(frac=1,random_state=0); F=['a','e','i_rad','varpi_sin','varpi_cos']
clf=RandomForestClassifier(100,max_depth=10,random_state=0).fit(tr[F],tr.label)
print('09 real avg P9 prob %.3f'%clf.predict_proba(r[F])[:,1].mean())
nulls=[]
for k in range(200):
    v=rng.uniform(0,2*np.pi,len(r)); X=r[F].copy(); X['varpi_sin']=np.sin(v); X['varpi_cos']=np.cos(v); nulls.append(clf.predict_proba(X)[:,1].mean())
print('09 avg prob on 200 uniform-random catalogs: median %.3f, 5-95%% %s'%(np.median(nulls),np.percentile(nulls,[5,95]).round(3)))
