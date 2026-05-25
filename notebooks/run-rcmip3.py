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
import copy
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import xarray as xr

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties
from tqdm.auto import tqdm

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
master_emissions = pd.read_csv(f"../data/{RCMIP3_VERSION}/RCMIP3_input_datafiles/rcmip_phase3_emissions_{RCMIP3_VERSION}.csv")
master_forcing = pd.read_csv(f"../data/{RCMIP3_VERSION}/RCMIP3_input_datafiles/rcmip_phase3_forcing_{RCMIP3_VERSION}.csv") 

master_cmip7_concentrations = pd.read_csv(f"../data/{RCMIP3_VERSION}/RCMIP3_input_datafiles/rcmip_phase3_concentrations_ScenarioMIP_{RCMIP3_VERSION}.csv")
master_cmip7_emissions = pd.read_csv(f"../data/{RCMIP3_VERSION}/RCMIP3_input_datafiles/rcmip_phase3_emissions_ScenarioMIP_{RCMIP3_VERSION}.csv")
master_cmip7_forcing = pd.read_csv(f"../data/{RCMIP3_VERSION}/RCMIP3_input_datafiles/rcmip_phase3_forcing_ScenarioMIP_{RCMIP3_VERSION}.csv") 

# %%
# variable mapping
variables_short = [var.split("|")[-1] for var in master_emissions.Variable.unique()] + [var.split("|")[-1] for var in master_forcing.Variable.unique()]
temp_dict = {var: var for var in variables_short}
for var in temp_dict:
    if var[:3]=='HFC':
        temp_dict[var] = f"{var[:3]}-{var[3:]}"
    elif var[:3]=='CFC':
        temp_dict[var] = f"{var[:3]}-{var[3:]}"
    elif var[:4]=='HCFC':
        temp_dict[var] = f"{var[:4]}-{var[4:]}"
    elif var[:5]=='Halon':
        temp_dict[var] = f"{var[:5]}-{var[5:]}"

temp_dict['cC4F8'] = "c-C4F8"
temp_dict["Energy and Industrial Processes"] = "CO2 FFI"
temp_dict["AFOLU"] = "CO2 AFOLU"
temp_dict["Land Use"] = "Land use"

RCMIP3_LOOKUP = {value: key for key, value in temp_dict.items()}
RCMIP3_LOOKUP["Albedo Change"] = "Land use"
#RCMIP3_LOOKUP

# %%
# unit dedafter
DEDAFTER = {specie: 1 for specie in RCMIP3_LOOKUP}
DEDAFTER["CO2 FFI"] = 0.001
DEDAFTER["CO2 AFOLU"] = 0.001
DEDAFTER["CO2"] = 0.001
DEDAFTER["N2O"] = 0.001

# %% [markdown]
# ## piControl
#
# Deviate slightly from protocol - only run with CO2, CH4 and N2O

# %%
scenario = "piControl"
exp_conc = "piControl"
exp_emis = None
startyear = 1750
endyear = 2500

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
        (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
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

# %% [markdown]
# ## esm-piControl
#
# As piControl, CO2 emissions driven

# %%
scenario = "esm-piControl"
exp_conc = "piControl"
exp_emis = "esm-piControl"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## esm-allGHG-piControl
#
# Run with CO2, CH4, N2O and all SLCFs (emissions driven), ignore minor GHGs?

# %%
scenario = "esm-allGHG-piControl"
exp_conc = None
exp_emis = "esm-piControl"
startyear = 1750
endyear = 2500

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
    if properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## 1pctCO2

# %%
scenario = "1pctCO2"
exp_conc = "1pctCO2"
exp_emis = None
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
        (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
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
# needed for 1pctCO2-bgc and 1pct-rad
forcing_1pctco2 = copy.deepcopy(f.forcing_sum)

# needed for branch experiments
cumulative_emissions_1pctco2 = copy.deepcopy(f.cumulative_emissions.sel(specie="CO2"))

# %% [markdown]
# ### esm-1pctCO2-brch-* sub-experiments
#
# We need to save out the state of every fair ensemble member upon hitting 750, 1000 or 2000 PgC as a restart for the branch experiments.
#
# So redo the run, 1 year at a time, checking to see if it exceeds 750, 1000 or 2000 PgC.

# %%
emissions_levels = [750, 1000, 2000]
f_iter = {}
gasbox_restarts = {}
temperature_restarts = {}
concentration_restarts = {}
forcing_restarts = {}
airborne_restarts = {}
cumulative_restarts = {}
alpha_restarts = {}

# %%
for cum_emis in emissions_levels:
    gasbox_restarts[cum_emis] = {}
    temperature_restarts[cum_emis] = {}
    concentration_restarts[cum_emis] = {}
    forcing_restarts[cum_emis] = {}
    airborne_restarts[cum_emis] = {}
    cumulative_restarts[cum_emis] = {}
    alpha_restarts[cum_emis] = {}

# %%
hit_cum_emis_df = {}
for cum_emis in emissions_levels:
    hit_cum_emis_df[cum_emis] = pd.Series([False]*len(valid_all), index=valid_all)

# %%
# set up a dummy fair runs for the purposes of using the structure for year[0]
for cum_emis in emissions_levels:
    f_iter[1849] = FAIR()
    f_iter[1849].define_time(1849, 1850, 1)
    f_iter[1849].define_scenarios(scenarios)
    f_iter[1849].define_species(species, properties)
    f_iter[1849].define_configs(valid_all)
    f_iter[1849].allocate()
    fill(f_iter[1849].forcing, 0)
    fill(f_iter[1849].temperature, 0)
    fill(f_iter[1849].airborne_emissions, 0)
    fill(f_iter[1849].cumulative_emissions, 0)
    fill(f_iter[1849].alpha_lifetime, 1)

# %%
for year in tqdm(range(1850, 1990)):
    f_iter[year] = FAIR()
    f_iter[year].define_time(year, year+1, 1)
    f_iter[year].define_scenarios(scenarios)
    f_iter[year].define_species(species, properties)
    f_iter[year].define_configs(valid_all)
    f_iter[year].allocate()

    for specie in f_iter[year].species:
        f_iter[year].concentration.loc[
            dict(
                timebounds=np.arange(year, year+2), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(year):str(year+1)
        ].T
    
    f_iter[year].fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
    f_iter[year].override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")
    
    # this needs to be undone from the override
    f_iter[year].species_configs["baseline_concentration"].loc[dict(specie="CO2")] = (
        df_defaults.loc[(df_defaults["name"]=="CO2"), "baseline_concentration"].values[0]
    )

    # also turn off stochasticity because we are running one year at a time
    fill(f_iter[year].climate_configs["stochastic_run"], False)
    fill(f_iter[year].climate_configs["use_seed"], True)
    
    initialise(f_iter[year].forcing, f_iter[year-1].forcing[-1, ...])
    initialise(f_iter[year].temperature, f_iter[year-1].temperature[-1, ...])
#    initialise(f_iter[year].concentration, f_iter[year-1].concentration[-1, ...]) 
    initialise(f_iter[year].airborne_emissions, f_iter[year-1].airborne_emissions[-1, ...])
    initialise(f_iter[year].cumulative_emissions, f_iter[year-1].cumulative_emissions[-1, ...])
    initialise(f_iter[year].alpha_lifetime, f_iter[year-1].alpha_lifetime[-1, ...])
    f_iter[year].gas_partitions=copy.deepcopy(f_iter[year-1].gas_partitions)
    
    f_iter[year].run(progress=False)
    
    # check if over 750, 1000, 2000 GtC
    for cum_emis in emissions_levels:
        for iconf, config in enumerate(f_iter[year].configs):
            if not hit_cum_emis_df[cum_emis].loc[config] and (f_iter[year].cumulative_emissions[-1, 0, iconf, 0] >= cum_emis*44.009/12.011):
                hit_cum_emis_df[cum_emis].loc[config] = True
                gasbox_restarts[cum_emis][config] = copy.deepcopy(f_iter[year].gas_partitions[0, iconf, :, :])
                forcing_restarts[cum_emis][config] = copy.deepcopy(f_iter[year].forcing[-1, 0, iconf, :])
                temperature_restarts[cum_emis][config] = copy.deepcopy(f_iter[year].temperature[-1, 0, iconf, :])
                concentration_restarts[cum_emis][config] = copy.deepcopy(f_iter[year].concentration[-1, 0, iconf, :])
                airborne_restarts[cum_emis][config] = copy.deepcopy(f_iter[year].airborne_emissions[-1, 0, iconf, :])
                cumulative_restarts[cum_emis][config] = copy.deepcopy(f_iter[year].cumulative_emissions[-1, 0, iconf, :])
                alpha_restarts[cum_emis][config] = copy.deepcopy(f_iter[year].alpha_lifetime[-1, 0, iconf, :])

# %% [markdown]
# ## 1pctCO2-4xext

# %%
scenario = "1pctCO2-4xext"
exp_conc = "1pctCO2-4xext"
exp_emis = None
startyear = 1750
endyear = 2500

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
        (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
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

# %% [markdown]
# ## 1pctCO2-cdr

# %%
scenario = "1pctCO2-cdr"
exp_conc = "1pctCO2-cdr"
exp_emis = None
startyear = 1750
endyear = 2500

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
        (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
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

# %% [markdown]
# ## 1pctCO2-bgc
#
# in RCMIP2, this was a hack acheived by introducing a fake forcing that was equal and opposite to the CO2 forcing, with the goal to ensure delta T = 0 and only the carbon cycle sees the CO2 increase.
#
# This could be acheived by using the 1pctCO2 forcing run and minusing this.
#
# It could also be achieved by trying to fix surface temperature as zero.

# %%
scenario = "1pctCO2-bgc"
exp_conc = "1pctCO2"
exp_emis = None
startyear = 1750
endyear = 1990

# %%
f = FAIR(ch4_method="Thornhill2021", temperature_prescribed=False)
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
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T

f.forcing.loc[dict(specie="minusCO2")] = -forcing_1pctco2.data

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

# this needs to be undone from the override
f.species_configs["baseline_concentration"].loc[dict(specie="CO2")] = df_defaults.loc[(df_defaults["name"]=="CO2"), "baseline_concentration"].values[0]

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

# # for BGC, set temperature to zero
# fill(f.temperature, 0)
# fill(f.forcing, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## 1pctCO2-rad
#
# I am not sure what the correct way of implementing this is, perhaps it would be to run in prescribed forcing mode with the forcing from 1pctCO2 and atmospheric CO2 fixed at pre-industrial. I don't think it would tell us anything since the separation in fair is technically cleanly done with 1pctCO2-bgc.

# %%
scenario = "1pctCO2-rad"
exp_conc = "piControl"
exp_emis = None
startyear = 1750
endyear = 1990

# %%
f = FAIR(ch4_method="Thornhill2021", temperature_prescribed=False)
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
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T

f.forcing.loc[dict(specie="plusCO2")] = forcing_1pctco2.data

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

# this needs to be undone from the override
f.species_configs["baseline_concentration"].loc[dict(specie="CO2")] = df_defaults.loc[(df_defaults["name"]=="CO2"), "baseline_concentration"].values[0]

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

# # for BGC, set temperature to zero
# fill(f.temperature, 0)
# fill(f.forcing, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## esm-1pct-brch-1000PgC

# %%
scenario = "esm-1pct-brch-1000PgC"
exp_conc = "piControl"
exp_emis = None
startyear = 1750  # not tracked
endyear = 2500

# %%
f = FAIR(ch4_method="Thornhill2021", temperature_prescribed=False)
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
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T

fill(f.emissions, 0, specie="CO2")

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

# this needs to be undone from the override
f.species_configs["baseline_concentration"].loc[dict(specie="CO2")] = df_defaults.loc[(df_defaults["name"]=="CO2"), "baseline_concentration"].values[0]

# set initial conditions
for iconf, config in enumerate(f.configs):
    initialise(f.concentration, concentration_restarts[1000][config], config=config)
    initialise(f.forcing, forcing_restarts[1000][config], config=config)
    initialise(f.temperature, temperature_restarts[1000][config], config=config)
    initialise(f.airborne_emissions, airborne_restarts[1000][config], config=config)
    initialise(f.cumulative_emissions, cumulative_restarts[1000][config], config=config)
    f.gas_partitions[0, iconf, :, :] = gasbox_restarts[1000][config]

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## esm-1pct-brch-2000PgC

# %%
scenario = "esm-1pct-brch-2000PgC"
exp_conc = "piControl"
exp_emis = None
startyear = 1750  # not tracked
endyear = 2500

# %%
f = FAIR(ch4_method="Thornhill2021", temperature_prescribed=False)
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
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T

fill(f.emissions, 0, specie="CO2")

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

# this needs to be undone from the override
f.species_configs["baseline_concentration"].loc[dict(specie="CO2")] = df_defaults.loc[(df_defaults["name"]=="CO2"), "baseline_concentration"].values[0]

# set initial conditions
for iconf, config in enumerate(f.configs):
    initialise(f.concentration, concentration_restarts[2000][config], config=config)
    initialise(f.forcing, forcing_restarts[2000][config], config=config)
    initialise(f.temperature, temperature_restarts[2000][config], config=config)
    initialise(f.airborne_emissions, airborne_restarts[2000][config], config=config)
    initialise(f.cumulative_emissions, cumulative_restarts[2000][config], config=config)
    f.gas_partitions[0, iconf, :, :] = gasbox_restarts[2000][config]

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## esm-1pct-brch-750PgC

# %%
scenario = "esm-1pct-brch-750PgC"
exp_conc = "piControl"
exp_emis = None
startyear = 1750  # not tracked
endyear = 2500

# %%
f = FAIR(ch4_method="Thornhill2021", temperature_prescribed=False)
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
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T

fill(f.emissions, 0, specie="CO2")

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

# this needs to be undone from the override
f.species_configs["baseline_concentration"].loc[dict(specie="CO2")] = df_defaults.loc[(df_defaults["name"]=="CO2"), "baseline_concentration"].values[0]

# set initial conditions
for iconf, config in enumerate(f.configs):
    initialise(f.concentration, concentration_restarts[750][config], config=config)
    initialise(f.forcing, forcing_restarts[750][config], config=config)
    initialise(f.temperature, temperature_restarts[750][config], config=config)
    initialise(f.airborne_emissions, airborne_restarts[750][config], config=config)
    initialise(f.cumulative_emissions, cumulative_restarts[750][config], config=config)
    f.gas_partitions[0, iconf, :, :] = gasbox_restarts[750][config]

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## abrupt-4xCO2

# %%
scenario = "abrupt-4xCO2"
exp_conc = "abrupt-4xCO2"
exp_emis = None
startyear = 1750
endyear = 2500

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
        (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
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

# %% [markdown]
# ## abrupt-2xCO2

# %%
scenario = "abrupt-2xCO2"
exp_conc = "abrupt-2xCO2"
exp_emis = None
startyear = 1750
endyear = 2500

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
        (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
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

# %% [markdown]
# ## abrupt-0p5xCO2

# %%
scenario = "abrupt-0p5xCO2"
exp_conc = "abrupt-0p5xCO2"
exp_emis = None
startyear = 1750
endyear = 2500

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
        (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
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

# %% [markdown]
# ## esm-pi-cdr-pulse

# %%
scenario = "esm-pi-cdr-pulse"
exp_conc = "piControl"
exp_emis = "esm-pi-cdr-pulse"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## esm-pi-CO2pulse

# %%
scenario = "esm-pi-CO2pulse"
exp_conc = "piControl"
exp_emis = "esm-pi-CO2pulse"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## esm-bell-1000PgC

# %%
scenario = "esm-bell-1000PgC"
exp_conc = "piControl"
exp_emis = "esm-bell-1000PgC"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## esm-bell-2000PgC

# %%
scenario = "esm-bell-2000PgC"
exp_conc = "piControl"
exp_emis = "esm-bell-2000PgC"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## esm-bell-750PgC

# %%
scenario = "esm-bell-750PgC"
exp_conc = "piControl"
exp_emis = "esm-bell-750PgC"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## historical
#
# here need to consider how to handle the 0.5 year offset
#
# it would be best to interpolate between successive years, repeating the 1750.0 value for 1749.0, and linearly extrapolating 2022.0 to 2023.0

# %%
scenario = "historical"
exp_conc = "historical"
exp_emis = "historical"
exp_forc = "historical"
startyear = 1750
endyear = 2023

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## historical-cmip6

# %%
scenario = "historical-cmip6"
exp_conc = "historical-cmip6"
exp_emis = "historical-cmip6"
exp_forc = "historical-cmip6"
startyear = 1750
endyear = 2015

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## hist-aer
#
# following Nathan Gillett's intent, we bin CO and VOC as aerosols rather than GHG. There's a slight inconsistency in the RCMIP input files so we take our source as the historical experiment.

# %%
scenario = "hist-aer"
exp_conc = None
exp_emis = "historical"  # consistency
exp_forc = None
startyear = 1750
endyear = 2023

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## hist-ghg
#
# again interpolate concentrations, and take the historical experiment as the source of concentration truth.

# %%
scenario = "hist-GHG"
exp_conc = "historical"  # consistency
exp_emis = None
exp_forc = None
startyear = 1750
endyear = 2023

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

# concentrations are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    working_concentrations = copy.deepcopy(
        master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T
    )
    working_concentrations.index = pd.to_numeric(working_concentrations.index)
    working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
    working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
    for midyear in np.arange(startyear-0.5, endyear):
        working_concentrations.loc[midyear] = np.nan
    working_concentrations = working_concentrations.sort_index()
    working_concentrations = working_concentrations.interpolate()
    f.concentration.loc[
        dict(
            timebounds=np.arange(startyear, endyear+1), 
            scenario=scenario,
            specie=specie
        )
    ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## hist-CO2

# %%
scenario = "hist-CO2"
exp_conc = "hist-CO2"  # I think leaving non-CO2 at 1850 is OK
exp_emis = None
exp_forc = None
startyear = 1750
endyear = 2023

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

# concentrations are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    working_concentrations = copy.deepcopy(
        master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T
    )
    working_concentrations.index = pd.to_numeric(working_concentrations.index)
    working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
    working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
    for midyear in np.arange(startyear-0.5, endyear):
        working_concentrations.loc[midyear] = np.nan
    working_concentrations = working_concentrations.sort_index()
    working_concentrations = working_concentrations.interpolate()
    f.concentration.loc[
        dict(
            timebounds=np.arange(startyear, endyear+1), 
            scenario=scenario,
            specie=specie
        )
    ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## ssp119

# %%
scenario = "ssp119"
exp_conc = "ssp119"
exp_emis = "ssp119"
exp_forc = "ssp119"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## ssp126

# %%
scenario = "ssp126"
exp_conc = "ssp126"
exp_emis = "ssp126"
exp_forc = "ssp126"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## ssp245

# %%
scenario = "ssp245"
exp_conc = "ssp245"
exp_emis = "ssp245"
exp_forc = "ssp245"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## ssp370

# %%
scenario = "ssp370"
exp_conc = "ssp370"
exp_emis = "ssp370"
exp_forc = "ssp370"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## ssp434

# %%
scenario = "ssp434"
exp_conc = "ssp434"
exp_emis = "ssp434"
exp_forc = "ssp434"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## ssp460

# %%
scenario = "ssp460"
exp_conc = "ssp460"
exp_emis = "ssp460"
exp_forc = "ssp460"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## ssp534-over

# %%
scenario = "ssp534-over"
exp_conc = "ssp534-over"
exp_emis = "ssp534-over"
exp_forc = "ssp534-over"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## ssp585

# %%
scenario = "ssp585"
exp_conc = "ssp585"
exp_emis = "ssp585"
exp_forc = "ssp585"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-hist

# %%
scenario = "esm-hist"
exp_conc = "historical"
exp_emis = "historical"
exp_forc = "historical"
startyear = 1750
endyear = 2023

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-hist-cmip6

# %%
scenario = "esm-hist-cmip6"
exp_conc = "historical-cmip6"
exp_emis = "historical-cmip6"
exp_forc = "historical-cmip6"
startyear = 1750
endyear = 2015

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-ssp119

# %%
scenario = "esm-ssp119"
exp_conc = "ssp119"
exp_emis = "ssp119"
exp_forc = "ssp119"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-ssp126

# %%
scenario = "esm-ssp126"
exp_conc = "ssp126"
exp_emis = "ssp126"
exp_forc = "ssp126"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-ssp245

# %%
scenario = "esm-ssp245"
exp_conc = "ssp245"
exp_emis = "ssp245"
exp_forc = "ssp245"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-ssp370

# %%
scenario = "esm-ssp370"
exp_conc = "ssp370"
exp_emis = "ssp370"
exp_forc = "ssp370"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-ssp434

# %%
scenario = "esm-ssp434"
exp_conc = "ssp434"
exp_emis = "ssp434"
exp_forc = "ssp434"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-ssp460

# %%
scenario = "esm-ssp460"
exp_conc = "ssp460"
exp_emis = "ssp460"
exp_forc = "ssp460"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-ssp534-over

# %%
scenario = "esm-ssp534-over"
exp_conc = "ssp534-over"
exp_emis = "ssp534-over"
exp_forc = "ssp534-over"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-ssp585

# %%
scenario = "esm-ssp585"
exp_conc = "ssp585"
exp_emis = "ssp585"
exp_forc = "ssp585"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-hist

# %%
scenario = "esm-allGHG-hist"
exp_conc = "historical"
exp_emis = "historical"
exp_forc = "historical"
startyear = 1750
endyear = 2023

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-hist-cmip6

# %%
scenario = "esm-allGHG-hist-cmip6"
exp_conc = "historical-cmip6"
exp_emis = "historical-cmip6"
exp_forc = "historical-cmip6"
startyear = 1750
endyear = 2015

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp119

# %%
scenario = "esm-allGHG-ssp119"
exp_conc = "ssp119"
exp_emis = "ssp119"
exp_forc = "ssp119"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp126

# %%
scenario = "esm-allGHG-ssp126"
exp_conc = "ssp126"
exp_emis = "ssp126"
exp_forc = "ssp126"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp245

# %%
scenario = "esm-allGHG-ssp245"
exp_conc = "ssp245"
exp_emis = "ssp245"
exp_forc = "ssp245"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp370

# %%
scenario = "esm-allGHG-ssp370"
exp_conc = "ssp370"
exp_emis = "ssp370"
exp_forc = "ssp370"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp370-lowNTCF

# %%
scenario = "esm-allGHG-ssp370-lowNTCF"
exp_conc = "esm-allGHG-ssp370-lowNTCF"
exp_emis = "esm-allGHG-ssp370-lowNTCF"
exp_forc = "ssp370"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp370-lowCH4

# %%
scenario = "esm-allGHG-ssp370-lowCH4"
exp_conc = "esm-allGHG-ssp370-lowCH4"
exp_emis = "esm-allGHG-ssp370-lowCH4"
exp_forc = "ssp370"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp370-lowNTCF-HighCH4

# %%
scenario = "esm-allGHG-ssp370-lowNTCF-HighCH4"
exp_conc = "esm-allGHG-ssp370-lowNTCF-HighCH4"
exp_emis = "esm-allGHG-ssp370-lowNTCF-HighCH4"
exp_forc = "ssp370"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp434

# %%
scenario = "esm-allGHG-ssp434"
exp_conc = "ssp434"
exp_emis = "ssp434"
exp_forc = "ssp434"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp460

# %%
scenario = "esm-allGHG-ssp460"
exp_conc = "ssp460"
exp_emis = "ssp460"
exp_forc = "ssp460"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp534-over

# %%
scenario = "esm-allGHG-ssp534-over"
exp_conc = "ssp534-over"
exp_emis = "ssp534-over"
exp_forc = "ssp534-over"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp534-over-highCH4

# %%
scenario = "esm-allGHG-ssp534-over-highCH4"
exp_conc = "esm-allGHG-ssp534-over-highCH4"
exp_emis = "esm-allGHG-ssp534-over-highCH4"
exp_forc = "ssp534-over"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp585

# %%
scenario = "esm-allGHG-ssp585"
exp_conc = "ssp585"
exp_emis = "ssp585"
exp_forc = "ssp585"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-ssp585-lowCH4

# %%
scenario = "esm-allGHG-ssp585-lowCH4"
exp_conc = "esm-allGHG-ssp585-lowCH4"
exp_emis = "esm-allGHG-ssp585-lowCH4"
exp_forc = "ssp585"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-scen7-H

# %%
scenario = "esm-scen7-H"
exp_conc = "scen7-H"
exp_emis = "scen7-H"
exp_forc = "scen7-H"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-scen7-HL

# %%
scenario = "esm-scen7-HL"
exp_conc = "scen7-HL"
exp_emis = "scen7-HL"
exp_forc = "scen7-HL"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-scen7-M

# %%
scenario = "esm-scen7-M"
exp_conc = "scen7-M"
exp_emis = "scen7-M"
exp_forc = "scen7-M"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-scen7-ML

# %%
scenario = "esm-scen7-ML"
exp_conc = "scen7-ML"
exp_emis = "scen7-ML"
exp_forc = "scen7-ML"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-scen7-L

# %%
scenario = "esm-scen7-L"
exp_conc = "scen7-L"
exp_emis = "scen7-L"
exp_forc = "scen7-L"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-scen7-VL

# %%
scenario = "esm-scen7-VL"
exp_conc = "scen7-VL"
exp_emis = "scen7-VL"
exp_forc = "scen7-VL"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-scen7-LN

# %%
scenario = "esm-scen7-LN"
exp_conc = "scen7-LN"
exp_emis = "scen7-LN"
exp_forc = "scen7-LN"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## scen7-HC

# %%
scenario = "scen7-HC"
exp_conc = "scen7-H"
exp_emis = "scen7-H"
exp_forc = "scen7-H"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## scen7-HLC

# %%
scenario = "scen7-HLC"
exp_conc = "scen7-HL"
exp_emis = "scen7-HL"
exp_forc = "scen7-HL"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## scen7-MC

# %%
scenario = "scen7-MC"
exp_conc = "scen7-M"
exp_emis = "scen7-M"
exp_forc = "scen7-M"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## scen7-MLC

# %%
scenario = "scen7-MLC"
exp_conc = "scen7-ML"
exp_emis = "scen7-ML"
exp_forc = "scen7-ML"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## scen7-LC

# %%
scenario = "scen7-LC"
exp_conc = "scen7-L"
exp_emis = "scen7-L"
exp_forc = "scen7-L"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## scen7-VLC

# %%
scenario = "scen7-VLC"
exp_conc = "scen7-VL"
exp_emis = "scen7-VL"
exp_forc = "scen7-VL"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## scen7-LNC

# %%
scenario = "scen7-LNC"
exp_conc = "scen7-LN"
exp_emis = "scen7-LN"
exp_forc = "scen7-LN"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-scen7-H

# %%
scenario = "esm-allGHG-scen7-H"
exp_conc = "scen7-H"
exp_emis = "scen7-H"
exp_forc = "scen7-H"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-scen7-HL

# %%
scenario = "esm-allGHG-scen7-HL"
exp_conc = "scen7-HL"
exp_emis = "scen7-HL"
exp_forc = "scen7-HL"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-scen7-H-CH4L

# %%
scenario = "esm-allGHG-scen7-H-CH4L"
exp_conc = "esm-allGHG-scen7-H-CH4L"
exp_emis = "esm-allGHG-scen7-H-CH4L"
exp_forc = "scen7-H"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-scen7-M

# %%
scenario = "esm-allGHG-scen7-M"
exp_conc = "scen7-M"
exp_emis = "scen7-M"
exp_forc = "scen7-M"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-scen7-ML

# %%
scenario = "esm-allGHG-scen7-ML"
exp_conc = "scen7-ML"
exp_emis = "scen7-ML"
exp_forc = "scen7-ML"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-scen7-L

# %%
scenario = "esm-allGHG-scen7-L"
exp_conc = "scen7-L"
exp_emis = "scen7-L"
exp_forc = "scen7-L"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-scen7-L-CH4H

# %%
scenario = "esm-allGHG-scen7-L-CH4H"
exp_conc = "esm-allGHG-scen7-L-CH4H"
exp_emis = "esm-allGHG-scen7-L-CH4H"
exp_forc = "scen7-L"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-scen7-VL

# %%
scenario = "esm-allGHG-scen7-VL"
exp_conc = "scen7-VL"
exp_emis = "scen7-VL"
exp_forc = "scen7-VL"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-allGHG-scen7-LN

# %%
scenario = "esm-allGHG-scen7-LN"
exp_conc = "scen7-LN"
exp_emis = "scen7-LN"
exp_forc = "scen7-LN"
startyear = 1750
endyear = 2500

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_cmip7_concentrations.loc[
                (master_cmip7_concentrations["Scenario"]==exp_conc) & (master_cmip7_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_cmip7_emissions.loc[
            (master_cmip7_emissions["Scenario"]==exp_emis) & (master_cmip7_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        forcing_scale = df_configs[f"forcing_scale[{specie}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_cmip7_forcing.loc[
                (master_cmip7_forcing["Scenario"]==exp_forc) & (master_cmip7_forcing["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## esm-flat10

# %%
scenario = "esm-flat10"
exp_conc = "piControl"
exp_emis = "esm-flat10"
startyear = 1750
endyear = 2170

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %% [markdown]
# ## esm-flat10-zec

# %%
scenario = "esm-flat10-zec"
exp_conc = "piControl"
exp_emis = "esm-flat10-zec"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat10-cdr

# %%
scenario = "esm-flat10-cdr"
exp_conc = "piControl"
exp_emis = "esm-flat10-cdr"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat10-nz

# %%
scenario = "esm-flat10-nz"
exp_conc = "piControl"
exp_emis = "esm-flat10-nz"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat10-rev

# %%
scenario = "esm-flat10-rev"
exp_conc = "piControl"
exp_emis = "esm-flat10-rev"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat7.5

# %%
scenario = "esm-flat7.5"
exp_conc = "piControl"
exp_emis = "esm-flat7.5"
startyear = 1750
endyear = 2170

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat7.5-zec

# %%
scenario = "esm-flat7.5-zec"
exp_conc = "piControl"
exp_emis = "esm-flat7.5-zec"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat7.5-cdr

# %%
scenario = "esm-flat7.5-cdr"
exp_conc = "piControl"
exp_emis = "esm-flat7.5-cdr"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat7.5-nz

# %%
scenario = "esm-flat7.5-nz"
exp_conc = "piControl"
exp_emis = "esm-flat7.5-nz"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat7.5-rev

# %%
scenario = "esm-flat7.5-rev"
exp_conc = "piControl"
exp_emis = "esm-flat7.5-rev"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat20

# %%
scenario = "esm-flat20"
exp_conc = "piControl"
exp_emis = "esm-flat20"
startyear = 1750
endyear = 2170

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat20-zec

# %%
scenario = "esm-flat20-zec"
exp_conc = "piControl"
exp_emis = "esm-flat20-zec"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat20-cdr

# %%
scenario = "esm-flat20-cdr"
exp_conc = "piControl"
exp_emis = "esm-flat20-cdr"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat20-nz

# %%
scenario = "esm-flat20-nz"
exp_conc = "piControl"
exp_emis = "esm-flat20-nz"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## esm-flat20-rev

# %%
scenario = "esm-flat20-rev"
exp_conc = "piControl"
exp_emis = "esm-flat20-rev"
startyear = 1750
endyear = 2500

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

# for idealised experiments only where there is a combination of emissions and concentrations sources, we will ignore the
# half year time offset and ignore the last time point of the emissions time series. In effect we add 0.5 to the time stamp
# in the emissions file.
for specie in f.species:
    if properties[specie]["input_mode"]=="concentration":
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = master_concentrations.loc[
            (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{specie}")),
            str(startyear):str(endyear)
        ].T
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ## methanemip-TM-allGHG

# %%
scenario = "methanemip-TM-allGHG"
exp_conc = None
exp_emis = "methanemip-TM-allGHG"
exp_forc = "ssp245"
startyear = 1750
endyear = 2130

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %% [markdown]
# ## methanemip-TM+BC-allGHG

# %%
scenario = "methanemip-TM+BC-allGHG"
exp_conc = None
exp_emis = "methanemip-TM+BC-allGHG"
exp_forc = "ssp245"
startyear = 1750
endyear = 2130

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

# concentrations and forcing are given as midyears
# so we want to subtract 0.5 years from the given value to put on to fair timebounds
for specie in f.species:
    if properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

f.fill_species_configs(f"../data/fair_calibration/{FAIR_CALIBRATION}/{scenario}/species_configs_properties.csv")
f.override_defaults(f"../data/fair_calibration/{FAIR_CALIBRATION}/calibrated_constrained_parameters.csv")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# save for later
f.to_netcdf(f"../output/native/{scenario}.nc")
(
    (f.alpha_lifetime.sel(specie=["CH4", "N2O"], scenario=scenario)) * f.species_configs["unperturbed_lifetime"].sel(specie=["CH4", "N2O"], gasbox=0)
).to_netcdf(f"../output/native/{scenario}_lifetimes.nc")

# %%
