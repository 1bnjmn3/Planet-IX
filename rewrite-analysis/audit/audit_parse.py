"""Audit check for ORIGINAL_PROJECT_AUDIT.md. Run from anywhere: python rewrite-analysis/audit/audit_parse.py"""
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
ORIG = str(_ROOT / "Benjamins code") + "/"
HELPERS = str(_ROOT / "Planet_IX" / "planetx_helpers.py")
import re, numpy as np

P=ORIG+'distant_extended.dat'
lines=open(P).read().splitlines()
fw=[]
for L in lines:
    if len(L)<103: continue
    try:
        fw.append(dict(id=L[0:7].strip(),M=float(L[26:35]),w=float(L[37:46]),O=float(L[48:57]),i=float(L[59:68]),e=float(L[70:79]),a=float(L[92:103]),name=L[166:194].strip()))
    except ValueError: pass
print('lines',len(lines),'fixed-width parsed',len(fw))
# 01 split parser
bad01=0; n01=0
fwd={f['id']:f for f in fw}
for L in lines:
    p=L.split()
    if len(L)<50 or len(p)<11: continue
    try: w,O,i,e,a=float(p[5]),float(p[6]),float(p[7]),float(p[8]),float(p[10])
    except: continue
    n01+=1
    f=fwd.get(L[0:7].strip())
    if f and (abs(f['a']-a)>1e-6 or abs(f['i']-i)>1e-6): bad01+=1
print('01 split parser rows',n01,'mismatched vs fixed width',bad01)
# helpers regex parser
from importlib import util
spec=util.spec_from_file_location('h',HELPERS); h=util.module_from_spec(spec); spec.loader.exec_module(h)
r=h.load_mpc_distant(P)
print('helpers regex parser used_rows',r['used_rows'],'of',r['raw_lines'])
# why dropped: check sample
drop=[]; wrong=0
for L in lines:
    nums=h._floats_from_line(L)
    if len(nums)<11: drop.append(L);continue
    f=fwd.get(L[0:7].strip())
    if f and abs(nums[10]-f['a'])>1e-6: wrong+=1
print('regex: lines with <11 floats',len(drop),' rows where nums[10]!=a',wrong)
print('example dropped/misaligned line:'); 
for L in lines[:400]:
    nums=h._floats_from_line(L); f=fwd.get(L[0:7].strip())
    if f and len(nums)>=11 and abs(nums[10]-f['a'])>1e-6: print(repr(L[:110])); print(nums[:12]); break
a=np.array([f['a'] for f in fw]); e=np.array([f['e'] for f in fw]); q=a*(1-e)
for (amin,qmin) in [(150,30),(230,30),(250,40)]:
    print(f'fixed-width cut a>{amin},q>{qmin}:',int(((a>amin)&(q>qmin)).sum()))
ra=r['a']; rq=ra*(1-r['e']); m=(ra>150)&(rq>30); print('regex-parser cut a>150,q>30:',int(m.sum()))
# varpi stats on both
def R(th): th=np.radians(th); return np.hypot(np.cos(th).mean(),np.sin(th).mean())
rng=np.random.default_rng(0)
def p(Rv,n,M=200000): th=rng.uniform(0,2*np.pi,(M,n)); return (np.hypot(np.cos(th).mean(1),np.sin(th).mean(1))>=Rv).mean()
vr=(r['Omega'][m]+r['omega'][m])%360; print('regex sample N',m.sum(),'R',R(vr),'p',p(R(vr),m.sum()))
O=np.array([f['O'] for f in fw]); w=np.array([f['w'] for f in fw])
mm=(a>150)&(q>30); v=(O[mm]+w[mm])%360; print('full sample a>150 q>30 N',mm.sum(),'R',R(v),'p',p(R(v),mm.sum()))
mm=(a>230)&(q>30); v=(O[mm]+w[mm])%360; print('full sample a>230 q>30 N',mm.sum(),'R',R(v),'p',p(R(v),mm.sum()))
print('----')
cnt=0
for L in lines:
    nums=h._floats_from_line(L); f=fwd.get(L[0:7].strip())
    if f and abs(nums[10]-f['a'])>1e-6:
        print(repr(L[:60])); print('  regex nums[3:11]',nums[3:11]); print('  true a',f['a']); cnt+=1
        if cnt==2: break
# which 20 objects did regex keep, and are their elements right?
ok=0; rows=[]
for L in lines:
    nums=h._floats_from_line(L)
    MM,ww,OO,ii,ee,aa=nums[4],nums[5],nums[6],nums[7],nums[8],nums[10]
    if not (aa>0 and 0<=ee<1.5 and 0<=ii<=180 and 0<=OO<=360 and 0<=ww<=360 and 0<=MM<=360): continue
    if aa>150 and aa*(1-ee)>30:
        f=fwd[L[0:7].strip()]; good=abs(f['a']-aa)<1e-6; ok+=good
        rows.append((f['name'] or f['id'],good))
print('regex sample of 20: correctly parsed',ok)
fwset=set(f['id'] for f in fw if f['a']>150 and f['a']*(1-f['e'])>30)
print('ids in true a>150,q>30 set:',len(fwset))
