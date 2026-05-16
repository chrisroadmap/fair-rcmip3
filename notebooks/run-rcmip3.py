# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Naively attempt to run rcmip3

# %%
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import xarray as xr

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

# %%
# setup
os.makedirs('../output/native', exist_ok=True)

# %%
# global defaults
RCMIP3_VERSION = "v1.1.7"
FAIR_CALIBRATION = "v1.6.1"

# %%
# common datasets
master_concentrations = pd.read_csv(f"../data/{RCMIP3_VERSION}/RCMIP3_input_datafiles/rcmip_phase3_concentrations_{RCMIP3_VERSION}.csv")

# %%
master_concentrations.Scenario.unique()

# %%
#master_concentrations.loc[master_concentrations["Scenario"]=="piControl"]

# %% [markdown]
# ## 1pctCO2

# %%
scenario = "1pctCO2"
startyear = 1750
endyear = 1990

# %%
f = FAIR(ch4_method="Thornhill2021")
scenarios = [scenario]

f.define_time(startyear, endyear, 1)
f.define_scenarios(scenarios)

species, properties = read_properties(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")

f.define_species(species, properties)
df_configs = pd.read_csv(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv", index_col=0)
df_defaults = pd.read_csv(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")

valid_all = df_configs.index

f.define_configs(valid_all)
f.allocate()

for specie in f.species:
    f.concentration.loc[
        dict(
            timebounds=np.arange(startyear, endyear+1), 
            scenario=scenario,
            specie=specie
        )
    ] = master_concentrations.loc[
        (master_concentrations["Scenario"]==scenario) & (master_concentrations["Variable"].str.endswith(specie)),
        str(startyear):str(endyear)
    ].T

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

# this needs to be undone from the override
f.species_configs["baseline_concentration"].loc[dict(specie="CO2")] = df_defaults.loc[(df_defaults["name"]=="CO2"), "baseline_concentration"].values[0]

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.airborne_fraction.sel(specie="CO2", scenario=scenario));

# %%
f.ghg_method

# %%
pl.plot(f.emissions.sel(scenario=scenario, specie="CO2")/3.664);

# %%
f.species_configs["baseline_concentration"]

# %%
f.to_netcdf('datadump.nc')

# %%
