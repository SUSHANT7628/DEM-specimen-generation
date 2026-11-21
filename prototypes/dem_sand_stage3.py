"""Density-control prototype built on the Stage 2 DEM concepts."""
from dem_sand_stage2 import *
SPECIMEN_WIDTH=.70; SPECIMEN_HEIGHT=1.20; E_MAX=.85; E_MIN=.55
TARGET_DENSITIES={'loose':.20,'medium':.50,'dense':.80}
def target_void_ratio(Dr): return E_MAX-Dr*(E_MAX-E_MIN)
def specimen_void_ratio(r):
    solid=np.sum(np.pi*r*r); total=SPECIMEN_WIDTH*SPECIMEN_HEIGHT; return (total-solid)/solid
def relative_density(e): return (E_MAX-e)/(E_MAX-E_MIN)
def generate_density_state(Dr):
    target=target_void_ratio(Dr); print(f'Target Dr={Dr:.2f}, target e={target:.4f}')
    return target
if __name__=='__main__':
    for name,Dr in TARGET_DENSITIES.items(): generate_density_state(Dr)
