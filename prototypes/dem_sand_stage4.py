"""Simplified triaxial-compression prototype using the Stage 2 DEM solver."""
import numpy as np
import matplotlib.pyplot as plt
from dem_sand_stage2 import generate_particles,calculate_forces,N,tangential_history,KN,CN,DAMPING,DT
SIGMA_3=50.; AXIAL_VELOCITY=.02; MAX_STEPS=100000; OUTPUT_INTERVAL=100

def specimen_dimensions(x,y,r): return np.max(x+r)-np.min(x-r),np.max(y+r)-np.min(y-r)
def calculate_wall_forces(x,y,r,v,top,bottom):
    top_force=bottom_force=0.
    for i in range(N):
        ob=r[i]-(y[i]-bottom)
        if ob>0: bottom_force+=max(KN*ob-CN*v[i,1],0)
        ot=y[i]+r[i]-top
        if ot>0: top_force+=max(KN*ot+CN*v[i,1],0)
    return top_force,bottom_force
def triaxial_test(x,y,r,m,v,max_strain=.20):
    global tangential_history; tangential_history={}; w0,h0=specimen_dimensions(x,y,r); bottom=0.; top=h0; out=[]
    for step in range(MAX_STEPS):
        f=calculate_forces(x,y,r,m,v); v+=f/m[:,None]*DT; x+=v[:,0]*DT; y+=v[:,1]*DT; top-=AXIAL_VELOCITY*DT
        for i in range(N):
            if y[i]+r[i]>top:y[i]=top-r[i]
        top_force,bottom_force=calculate_wall_forces(x,y,r,v,top,bottom); w,h=specimen_dimensions(x,y,r); sigma1=top_force/max(w,1e-12); q=sigma1-SIGMA_3; strain=(h0-h)/h0
        if step%OUTPUT_INTERVAL==0:out.append((strain,sigma1,q,w,h))
        v*=DAMPING
        if strain>=max_strain:break
    return np.asarray(out)
def friction_angle(peak_q,sigma3=SIGMA_3):
    ratio=peak_q/sigma3; return np.degrees(np.arcsin(ratio/(2+ratio)))
def plot(results):
    plt.figure(figsize=(8,5));plt.plot(results[:,0]*100,results[:,2]);plt.xlabel('Axial strain (%)');plt.ylabel('Deviator stress');plt.title('Simplified DEM Triaxial Compression');plt.grid();plt.show()
if __name__=='__main__':
    x,y,r,m,v=generate_particles();res=triaxial_test(x,y,r,m,v);plot(res);peak=res[np.argmax(res[:,2]),2];print(f'Peak q={peak:.3f}; estimated friction angle={friction_angle(peak):.2f} deg')
