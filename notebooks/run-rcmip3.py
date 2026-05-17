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

# %%
# variable mapping
variables_short = [var.split("|")[-1] for var in master_emissions.Variable.unique()]
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

RCMIP3_LOOKUP = {value: key for key, value in temp_dict.items()}

# %%
#RCMIP3_LOOKUP

# %%
# unit dedafter
DEDAFTER = {specie: 1 for specie in RCMIP3_LOOKUP}
DEDAFTER["CO2 FFI"] = 0.001
DEDAFTER["CO2 AFOLU"] = 0.001
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

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %%
pl.plot(f.forcing_sum.sel(scenario=scenario));

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

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %%
pl.plot(f.forcing_sum.sel(scenario=scenario));

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

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %%
pl.plot(f.forcing_sum.sel(scenario=scenario));

# %%
pl.plot(f.forcing.sel(scenario=scenario, specie="CO2"));

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

# %%
pl.hist(cumulative_emissions_1pctco2.sel(timebounds=1990, scenario=scenario)/3.664)

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %% [markdown]
# ### 1pctCO2 sub-experiment
#
# We need to save out the state of every fair ensemble member upon hitting 1000 PgC as a restart for the branch experiments.
#
# So redo the run, 1 year at a time, checking to see if it exceeds 1000 PgC.

# %%
f_iter = {}
gasbox_restarts = {}
temperature_restarts = {}
forcing_restarts = {}
airborne_restarts = {}
cumulative_restarts = {}
alpha_restarts = {}

# %%
hit1000_df = pd.Series([False]*len(valid_all), index=valid_all)
hit1000_df

# %%
# set up a dummy fair run for the purposes of using the structure for year[0]
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
### START HERE NEXT TIME

for year in tqdm(range(140)):
    f_iter[year] = FAIR()
    f_iter[cal][year].define_time(year, year+1, 1)
        f_iter[cal][year].define_scenarios(scenarios)
        f_iter[cal][year].define_species(species, properties)
        f_iter[cal][year].define_configs(list(cal_df[cal].index))
        f_iter[cal][year].allocate()
    
        f_iter[cal][year].concentration.loc[dict(specie='CO2')] = conc_df.values[year:year+2,None]
        f_iter[cal][year].concentration.loc[dict(specie='CH4')] = 808.2490285
        f_iter[cal][year].concentration.loc[dict(specie='N2O')] = 273.021047
        
        # Get default species configs
        f_iter[cal][year].fill_species_configs()
        
        # climate response
        fill(f_iter[cal][year].climate_configs["ocean_heat_capacity"], cal_df[cal].loc[:, "clim_c1":"clim_c3"].values)
        fill(
            f_iter[cal][year].climate_configs["ocean_heat_transfer"],
            cal_df[cal].loc[:, "clim_kappa1":"clim_kappa3"].values,
        )
        fill(f_iter[cal][year].climate_configs["deep_ocean_efficacy"], cal_df[cal].loc[:, "clim_epsilon"])
        fill(f_iter[cal][year].climate_configs["gamma_autocorrelation"], cal_df[cal].loc[:, "clim_gamma"])
        fill(f_iter[cal][year].climate_configs["sigma_eta"], cal_df[cal].loc[:, "clim_sigma_eta"])
        fill(f_iter[cal][year].climate_configs["sigma_xi"], cal_df[cal].loc[:, "clim_sigma_xi"])
        fill(f_iter[cal][year].climate_configs["seed"], cal_df[cal].loc[:, "seed"])
        fill(f_iter[cal][year].climate_configs["stochastic_run"], False)
        fill(f_iter[cal][year].climate_configs["use_seed"], True)
        fill(f_iter[cal][year].climate_configs["forcing_4co2"], cal_df[cal].loc[:, "clim_F_4xCO2"])
    
        # carbon cycle
        fill(f_iter[cal][year].species_configs["iirf_0"], cal_df[cal].loc[:,"cc_r0"], specie="CO2")
        fill(
            f_iter[cal][year].species_configs["iirf_airborne"], cal_df[cal].loc[:,"cc_rA"], specie="CO2"
        )
        fill(f_iter[cal][year].species_configs["iirf_uptake"], cal_df[cal].loc[:,"cc_rU"], specie="CO2")
        fill(
            f_iter[cal][year].species_configs["iirf_temperature"],
            cal_df[cal].loc[:,"cc_rT"],
            specie="CO2",
        )
    
        # forcing scaling
        fill(
            f_iter[cal][year].species_configs["forcing_scale"],
            cal_df[cal].loc[:, "fscale_CO2"],
            specie="CO2",
        )
    
        # initial condition of CO2 concentration (but not baseline for forcing calculations)
        fill(
            f_iter[cal][year].species_configs["baseline_concentration"],
            cal_df[cal].loc[:,"cc_co2_concentration_1750"],
            specie="CO2",
        )
        fill(f_iter[cal][year].species_configs['baseline_concentration'], 808.2490285, specie='CH4')
        fill(f_iter[cal][year].species_configs['baseline_concentration'], 273.021047, specie='N2O')
        
        initialise(f_iter[cal][year].forcing, f_iter[cal][year-1].forcing[-1, ...])
        initialise(f_iter[cal][year].temperature, f_iter[cal][year-1].temperature[-1, ...])
        initialise(f_iter[cal][year].airborne_emissions, f_iter[cal][year-1].airborne_emissions[-1, ...])
        initialise(f_iter[cal][year].cumulative_emissions, f_iter[cal][year-1].cumulative_emissions[-1, ...])
        initialise(f_iter[cal][year].alpha_lifetime, f_iter[cal][year-1].alpha_lifetime[-1, ...])
        f_iter[cal][year].gas_partitions=copy.deepcopy(f_iter[cal][year-1].gas_partitions)
    
        # do the run
        f_iter[cal][year].run(progress=False)
    
        # check if over 1000 GtC
        for iconf, config in enumerate(f_iter[cal][year].configs):
            if not hit1000_df[cal].loc[config] and (f_iter[cal][year].cumulative_emissions[-1, 0, iconf, 0] >= 1000*44.009/12.011):
                hit1000_df[cal].loc[config] = True
                gasbox_restarts[cal][config] = copy.deepcopy(f_iter[cal][year].gas_partitions[0, iconf, :, :])
                forcing_restarts[cal][config] = copy.deepcopy(f_iter[cal][year].forcing[-1, 0, iconf, :])
                temperature_restarts[cal][config] = copy.deepcopy(f_iter[cal][year].temperature[-1, 0, iconf, :])
                airborne_restarts[cal][config] = copy.deepcopy(f_iter[cal][year].airborne_emissions[-1, 0, iconf, :])
                cumulative_restarts[cal][config] = copy.deepcopy(f_iter[cal][year].cumulative_emissions[-1, 0, iconf, :])
                alpha_restarts[cal][config] = copy.deepcopy(f_iter[cal][year].alpha_lifetime[-1, 0, iconf, :])

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

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

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

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

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

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

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

# %%
pl.plot(f.temperature.sel(layer=0, scenario=scenario));

# %%
