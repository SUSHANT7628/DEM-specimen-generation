import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
N=400; WIDTH=1.; HEIGHT=2.; D_MIN=.020; D_MAX=.055; DENSITY=2500.; KN=2e4; KT=8e3; CN=2.; MU=.45; GRAVITY=9.81; DT=5e-5; MAX_STEPS=50000; DAMPING=.9995; ENERGY_TOL=1e-7
tangential_history={}
def generate_psd(): return np.clip(np.random.lognormal(np.log(.032),.25,N),D_MIN,D_MAX)
def generate_particles():
    r=generate_psd()/2; x=np.random.uniform(D_MAX,WIDTH-D_MAX,N); y=np.random.uniform(HEIGHT*.55,HEIGHT-D_MAX,N); m=DENSITY*np.pi*r*r; return x,y,r,m,np.zeros((N,2))
def contact_force(i,j,x,y,r,v):
    dvec=np.array([x[i]-x[j],y[i]-y[j]]); d=np.linalg.norm(dvec); ov=r[i]+r[j]-d
    if ov<=0:return np.zeros(2),None
    n=dvec/d if d>1e-12 else np.array([1.,0.]); t=np.array([-n[1],n[0]]); rv=v[i]-v[j]; vn=rv@n; vt=rv@t; fn=max(KN*ov-CN*vn,0); key=(i,j); tangential_history.setdefault(key,0.); tangential_history[key]+=vt*DT; ft=-KT*tangential_history[key]; lim=MU*fn
    if abs(ft)>lim: ft=-np.sign(ft)*lim; tangential_history[key]=-ft/KT
    return fn*n+ft*t,key
def calculate_forces(x,y,r,m,v):
    f=np.zeros((N,2)); f[:,1]-=m*GRAVITY
    for i in range(N):
        for j in range(i+1,N):
            ff,key=contact_force(i,j,x,y,r,v)
            if key is not None:f[i]+=ff; f[j]-=ff
    for i in range(N):
        vals=[(0,r[i]-x[i],1),(0,x[i]+r[i]-WIDTH,-1),(1,r[i]-y[i],1),(1,y[i]+r[i]-HEIGHT,-1)]
        for a,ov,s in vals:
            if ov>0:f[i,a]+=s*KN*ov
    return f
def run_dem():
    global tangential_history; tangential_history={}; x,y,r,m,v=generate_particles()
    for step in range(MAX_STEPS):
        f=calculate_forces(x,y,r,m,v); v+=f/m[:,None]*DT; x+=v[:,0]*DT; y+=v[:,1]*DT; v*=DAMPING
        ke=.5*np.sum(m*np.sum(v*v,axis=1))
        if step%2000==0: print(f'Step {step:6d} | KE={ke:.4e}')
        if ke<ENERGY_TOL: break
    return x,y,r
def calculate_void_ratio(r):
    solid=np.sum(np.pi*r*r); total=WIDTH*HEIGHT; void=total-solid; return void/solid,void/total
def plot_specimen(x,y,r):
    fig,ax=plt.subplots(figsize=(7,10))
    for i in range(N):ax.add_patch(plt.Circle((x[i],y[i]),r[i],fill=False,linewidth=.7))
    ax.set(xlim=(0,WIDTH),ylim=(0,HEIGHT),aspect='equal',title='DEM Sand Specimen',xlabel='x',ylabel='y');plt.show()
if __name__=='__main__':
    x,y,r=run_dem();e,n=calculate_void_ratio(r);print(f'Void ratio={e:.4f}\nPorosity={n:.4f}');plot_specimen(x,y,r)
