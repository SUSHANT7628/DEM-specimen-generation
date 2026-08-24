# Force-Based DEM Sand Specimen Generator

A Python educational DEM project for generating and mechanically loading 2-D sand specimens.

## Contents
- `dem_sand_stage1.py` — basic force-based particle settling
- `dem_sand_stage2.py` — PSD + frictional contact model
- `dem_sand_stage3.py` — density-controlled specimen workflow
- `dem_sand_stage4.py` — simplified triaxial compression workflow
- `requirements.txt` — Python dependencies

## Run
```bash
pip install -r requirements.txt
python dem_sand_stage1.py
python dem_sand_stage2.py
python dem_sand_stage3.py
python dem_sand_stage4.py
```

## Note
These scripts are educational/research prototypes, not validated commercial-grade DEM software. Material parameters and boundary conditions should be calibrated against experimental data before using results for engineering decisions.
