import numpy as np
import matplotlib.pyplot as plt
N=250; WIDTH=1.0; HEIGHT=2.0; R_MIN=0.015; R_MAX=0.030
DENSITY=2500.0; KN=2e4; CN=2.0; GRAVITY=9.81; DT=1e-4; MAX_STEPS=30000

def generate_particles():
    radii=np.random.uniform(R_MIN,R_MAX,N); x=np.random.uniform(R_MIN,WIDTH-R_MIN,N); y=np.random.uniform(HEIGHT*.5,HEIGHT-R_MAX,N)
    mass=DENSITY*np.pi*radii**2; velocity=np.zeros((N,2)); return x,y,radii,mass,velocity

def calculate_forces(x,y,radii,mass,velocity):
    forces=np.zeros((N,2)); forces[:,1]-=mass*GRAVITY
    for i in range(N):
        for j in range(i+1,N):
            dvec=np.array([x[i]-x[j],y[i]-y[j]]); d=np.linalg.norm(dvec); overlap=radii[i]+radii[j]-d
            if overlap>0:
                normal=dvec/d if d>1e-12 else np.array([1.,0.]); vn=np.dot(velocity[i]-velocity[j],normal)
                fn=max(KN*overlap-CN*vn,0); f=fn*normal; forces[i]+=f; forces[j]-=f
    for i in range(N):
        for axis,overlap in [(0,radii[i]-x[i]),(0,x[i]+radii[i]-WIDTH),(1,radii[i]-y[i]),(1,y[i]+radii[i]-HEIGHT)]:
            if overlap>0:
                sign=1 if (axis==0 and overlap==radii[i]-x[i]) or (axis==1 and overlap==radii[i]-y[i]) else -1
                forces[i,axis]+=sign*KN*overlap
    return forces

def run_dem():
    x,y,r,m,v=generate_particles()
    for step in range(MAX_STEPS):
        f=calculate_forces(x,y,r,m,v); a=f/m[:,None]; v+=a*DT; x+=v[:,0]*DT; y+=v[:,1]*DT; v*=.999
        if step%1000==0: print(f'Step {step:6d} | KE={.5*np.sum(m*np.sum(v*v,axis=1)):.6e}')
    return x,y,r

def plot_specimen(x,y,r):
    fig,ax=plt.subplots(figsize=(6,10))
    for i in range(N): ax.add_patch(plt.Circle((x[i],y[i]),r[i],fill=False,linewidth=.8))
    ax.set(xlim=(0,WIDTH),ylim=(0,HEIGHT),aspect='equal',title='Force-Based DEM Sand Specimen',xlabel='x',ylabel='y'); plt.show()
if __name__=='__main__': plot_specimen(*run_dem())
