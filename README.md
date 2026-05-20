# fair simulations for rcmip phase 3

- fair version 2.2.4
- calibration version 1.6.1
- rcmip bundle version 1.1.7

## to get started

1. clone the repository
2. create the conda environment with `conda env create -f environment.yml`
3. activate with `conda activate fair-rcmip3`
4. grab the RCMIP dataset zipfile from https://doi.org/10.5281/zenodo.20154638 and save it in the `data` folder 
5. `cd data` and `unzip RCMIP3_protocol_bundle_V1_1_7.zip -d v1.1.7` (replace the filename and version if necessary)
6. grab the super-duper top-secret CMIP7 scenarios from https://gitlab.com/rcmip/rcmip-phase-3-scenariomip/ and download them into `data/v*/RCMIP3_input_files/`
7. `jupyter notebook` then run the scripts in the `scripts` folder
