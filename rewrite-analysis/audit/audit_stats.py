"""Audit check for ORIGINAL_PROJECT_AUDIT.md. Run from anywhere: python rewrite-analysis/audit/audit_stats.py"""
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
ORIG = str(_ROOT / "Benjamins code") + "/"
HELPERS = str(_ROOT / "Planet_IX" / "planetx_helpers.py")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
D=ORIG
sm=pd.read_csv(D+'smart_dataset.csv'); live=pd.read_csv(D+'live_mpc_data.csv')
rng=np.random.default_rng(0)
print('smart N',len(sm),'i median',sm.i.median(),'mean',sm.i.mean(), 'retrograde',(sm.i>90).sum())
def peak(i):
    k=gaussian_kde(i); x=np.linspace(0,60,200); return x[np.argmax(k(x))]
print('29 peak',peak(sm.i))
# bootstrap the peak
bp=[peak(rng.choice(sm.i,len(sm))) for _ in range(1000)]; print('29 peak bootstrap 16-84',np.percentile(bp,[2.5,16,50,84,97.5]))
# null: isotropic nodes, sin(i)*gaussian sigma s
for s in [12,15]:
    pk=[]
    for _ in range(500):
        u=[]
        while len(u)<len(sm):
            x=abs(rng.normal(0,s)); 
            if rng.random()<np.sin(np.radians(x)) : u.append(x)
        pk.append(peak(np.array(u)))
    print(f'null sin(i)*N(0,{s}) peaks: median',np.median(pk),'16-84',np.percentile(pk,[16,84]))
# proper mean plane: mean angular momentum unit vector
def poles(i,O):
    i=np.radians(i);O=np.radians(O); return np.c_[np.sin(i)*np.sin(O),-np.sin(i)*np.cos(O),np.cos(i)]
def meanplane(i,O):
    h=poles(i,O).sum(0); h/=np.linalg.norm(h); inc=np.degrees(np.arccos(h[2])); node=np.degrees(np.arctan2(h[0],-h[1]))%360; return inc,node
pro=sm[sm.i<90]
for name,d in [('smart all',sm),('smart prograde',pro)]:
    inc,node=meanplane(d.i,d.Node); bs=[meanplane(*d.sample(len(d),replace=True,random_state=k)[['i','Node']].values.T) for k in range(2000)]
    bi=np.array([b[0] for b in bs]); print(f'{name} N={len(d)} mean-plane tilt {inc:.1f} deg node {node:.1f}; bootstrap tilt 16/50/84 {np.percentile(bi,[16,50,84])}')
    # null: same i, random nodes
    nt=[meanplane(d.i.values,rng.uniform(0,360,len(d)))[0] for _ in range(5000)]
    print('   null(random nodes) tilt median',np.median(nt),' p(tilt>=obs)',np.mean(np.array(nt)>=inc))
# 24/28 bootstrap "confidence" under a null
def conf_28(df,trials=200,window=(12,25),ncomp=None,bic_range=None):
    X=pd.DataFrame({'i':df.i.values,'s':np.sin(np.radians(df.Node.values)),'c':np.cos(np.radians(df.Node.values))})
    sc=StandardScaler().fit(X); ok=0
    for k in range(trials):
        Xr=sc.transform(X.sample(len(X),replace=True,random_state=k))
        if bic_range:
            best=min((GaussianMixture(n,random_state=k).fit(Xr) for n in bic_range),key=lambda g:g.bic(Xr))
        else: best=GaussianMixture(ncomp,random_state=k).fit(Xr)
        mi=best.means_[:,0]*sc.scale_[0]+sc.mean_[0]; ok+=np.any((mi>window[0])&(mi<window[1]))
    return ok/trials
st=live[(live.a>250)&(live.q>40)]
print('24 real (N=%d) conf'%len(st),conf_28(st,200,(12,22),bic_range=range(1,4)))
null=pd.DataFrame({'i':rng.permutation(st.i.values),'Node':rng.uniform(0,360,len(st))})
print('24 on NULL (same i, random node) conf',conf_28(null,200,(12,22),bic_range=range(1,4)))
print('24: mean i of stable set',st.i.mean(),' -> 1-component GMM mean lies in window trivially')
# 21: GMM BIC with null
def bestn(df,seed=42):
    X=StandardScaler().fit_transform(np.c_[df.i,np.sin(np.radians(df.Node)),np.cos(np.radians(df.Node))])
    b=[GaussianMixture(n,random_state=seed,n_init=10).fit(X).bic(X) for n in range(1,6)]; return int(np.argmin(b))+1,b
print('21 real',bestn(st))
cnt=[bestn(pd.DataFrame({'i':rng.permutation(st.i.values),'Node':rng.uniform(0,360,len(st))}))[0] for _ in range(50)]
print('21 on 50 NULL catalogs (random nodes): best n counts',np.bincount(cnt,minlength=6)[1:])
# 18: DBSCAN eps sensitivity + arithmetic node mean
hq=live[(live.a>150)&(live.q>30)]
X=StandardScaler().fit_transform(np.c_[hq.i,np.sin(np.radians(hq.Node)),np.cos(np.radians(hq.Node))])
for eps in [0.3,0.4,0.5,0.6,0.7]:
    l=DBSCAN(eps=eps,min_samples=4).fit(X).labels_; n=len(set(l))-(-1 in l)
    if n:
        top=pd.Series(l[l!=-1]).mode()[0]; c=hq[l==top]
        cm=np.degrees(np.arctan2(np.sin(np.radians(c.Node)).mean(),np.cos(np.radians(c.Node)).mean()))%360
        print(f'18 eps={eps}: clusters={n} top N={len(c)} i={c.i.mean():.1f} node(arith)={c.Node.mean():.1f} node(circ)={cm:.1f}')
    else: print(f'18 eps={eps}: clusters=0')
l=DBSCAN(eps=0.5,min_samples=4).fit(X).labels_; top=pd.Series(l[l!=-1]).mode()[0]; c=hq[l==top]; print('18 top cluster nodes',sorted(c.Node.round(0).tolist()))
# 31 silhouette driven by retrograde
from sklearn.cluster import AgglomerativeClustering; from sklearn.metrics import silhouette_score
def sil(d):
    F=pd.DataFrame({'a':d.a,'e':d.e,'i':d.i,'ns':np.sin(np.radians(d.Node)),'nc':np.cos(np.radians(d.Node)),'ps':np.sin(np.radians(d.w)),'pc':np.cos(np.radians(d.w))})
    X=StandardScaler().fit_transform(F); return max((silhouette_score(X,AgglomerativeClustering(k).fit_predict(X)),k) for k in range(2,6))
print('31 all',sil(sm),' without 2 retrograde',sil(sm[sm.i<90]))
print('retrograde objs',sm[sm.i>90][['ID','a','i']].to_string(index=False))
