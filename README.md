# fair simulations for rcmip phase 3

- fair version 2.2.4
- calibration version 1.6.1
- rcmip bundle version 1.1.7

## to run the protocol

1. clone the repository
2. create the conda environment with `conda env create -f environment.yml`
3. activate with `conda activate fair-rcmip3`
4. grab the RCMIP dataset zipfile from https://doi.org/10.5281/zenodo.20154638 and save it in the `data` folder 
5. `cd data` and `unzip RCMIP3_protocol_bundle_V1_1_7.zip -d v1.1.7` (replace the filename and version if necessary)
6. grab the super-duper top-secret CMIP7 scenarios from https://gitlab.com/rcmip/rcmip-phase-3-scenariomip/ and download them into `data/v*/RCMIP3_input_files/`
7. `jupyter notebook` then run the scripts in the `scripts` folder in order

## to upload results

1. First check to see that all of the validation works on the output files:
  
```
cd output
rcmip validate --comments rcmip3_model_comments.csv pyrcmip/*.nc rcmip3_model_metadata.csv
```

2. Request a token for upload from Alex Romero-Prieto (see https://pyrcmip.readthedocs.io/en/latest/cli.html#rcmip-upload)
3. Choose a semantic version for the upload - this will not be the same as the model or the calibration version.
4. do the upload with `rcmip upload --token TOKEN --model fair --version VERSION --comments rcmip3_model_comments.csv pyrcmip/*.nc rcmip3_model_metadata.csv`