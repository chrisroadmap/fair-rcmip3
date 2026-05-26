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
# # Convert output from fair to rcmip format

# %%
import os

from fair.earth_params import molecular_weight_air, mass_atmosphere, earth_radius, seconds_per_year
import numpy as np
import pandas as pd
import xarray as xr
from scmdata.run import ScmRun, run_append
from scmdata.netcdf import nc_to_run

# %%
os.makedirs("../output/pyrcmip", exist_ok=True)

# %%
CO2_MASS = 44.009
C_MASS = 12.011

# %%
output_variables = list(pd.read_excel("../data/v1.1.7/submission templates/rcmip3_model_output_test.xlsx", sheet_name="variable_definitions").Variable.unique())

# %%
output_variables

# %%
# variable mapping
variables_short = list(pd.Series([var.split("|")[-1] for var in output_variables]).unique())
variable_mapping = {var: var for var in variables_short}
for var in variable_mapping:
    if var[:3]=='HFC':
        variable_mapping[var] = f"{var[:3]}-{var[3:]}"
    elif var[:3]=='CFC':
        variable_mapping[var] = f"{var[:3]}-{var[3:]}"
    elif var[:4]=='HCFC':
        variable_mapping[var] = f"{var[:4]}-{var[4:]}"
    elif var[:5]=='Halon':
        variable_mapping[var] = f"{var[:5]}-{var[5:]}"

variable_mapping['cC4F8'] = "c-C4F8"

# %%
concentrations_variable_mapping = {variable_mapping[var.split('|')[-1]]:var for var in output_variables if var.startswith('Atmospheric Concentrations')}
del concentrations_variable_mapping["Halon-1202"]

# %%
concentrations_unit_mapping = {var: 'ppt' for var in concentrations_variable_mapping}
concentrations_unit_mapping['CO2'] = 'ppm'
concentrations_unit_mapping['CH4'] = 'ppb'
concentrations_unit_mapping['N2O'] = 'ppb'

# %%
lifetime_variable_mapping = {var.split('|')[-1]:var for var in output_variables if var.startswith('Atmospheric Lifetime')}
lifetime_variable_mapping

# %%
forcing_variable_mapping = {variable_mapping[var.split('|')[-1]]:var for var in output_variables if var.startswith('Effective Radiative Forcing')}
del forcing_variable_mapping["BC"]
del forcing_variable_mapping["Biomass Burning"]
del forcing_variable_mapping["NH3"]
del forcing_variable_mapping["Nitrate"]
del forcing_variable_mapping["OC"]
del forcing_variable_mapping["Sulfate"]
del forcing_variable_mapping["Fossil and Industrial"]
del forcing_variable_mapping["Mineral Dust"]
del forcing_variable_mapping["Stratospheric Contribution"]
del forcing_variable_mapping["Tropospheric Contribution"]
del forcing_variable_mapping["Halon-1202"]

del forcing_variable_mapping["Effective Radiative Forcing"] # class, add manually
del forcing_variable_mapping["Anthropogenic"] # class, add manually
del forcing_variable_mapping["Aerosol"] # class, add manually
del forcing_variable_mapping["Other"] # class, add manually
del forcing_variable_mapping["F-Gases"] # class, add manually
del forcing_variable_mapping["Montreal Gases"] # class, add manually
del forcing_variable_mapping["Other WMGHGs"] # class, add manually
del forcing_variable_mapping["Natural"] # class, add manually
del forcing_variable_mapping["HFC-"] # class, add manually
del forcing_variable_mapping["CFC-"] # class, add manually
del forcing_variable_mapping["PFC"] # class, add manually

forcing_variable_mapping["Light absorbing particles on snow and ice"] = forcing_variable_mapping.pop("BC on Snow")
forcing_variable_mapping["Stratospheric water vapour"] = forcing_variable_mapping.pop("Stratospheric H2O")
forcing_variable_mapping["Aerosol-cloud interactions"] = forcing_variable_mapping.pop("Aerosol-cloud Interactions")
forcing_variable_mapping["Aerosol-radiation interactions"] = forcing_variable_mapping.pop("Aerosol-radiation Interactions")

# %%
#carbonpool_variable_mapping = {"CO2": "Carbon Pool|Atmosphere"}

# %%
emissions_variable_mapping = {variable_mapping[var.split('|')[-1]]:var for var in output_variables if var.startswith('Emissions')}
del emissions_variable_mapping["Biomass Burning"]
del emissions_variable_mapping["Fossil, Industry and AFOLU"]
del emissions_variable_mapping["Other"]
del emissions_variable_mapping["Fossil and Industry"]
del emissions_variable_mapping["Land Use Change"]
del emissions_variable_mapping["BC"]
del emissions_variable_mapping["OC"]
del emissions_variable_mapping["Sulfur"]
del emissions_variable_mapping["CO"]
del emissions_variable_mapping["VOC"]
del emissions_variable_mapping["NOx"]
del emissions_variable_mapping["NH3"]

# %%
emissions_variable_mapping

# %%
emissions_unit_mapping = {var: f'kt {emissions_variable_mapping[var].split("|")[-1]}/yr' for var in emissions_variable_mapping}
emissions_unit_mapping['CO2'] = 'Gt CO2/yr'
emissions_unit_mapping['CH4'] = 'Mt CH4/yr'
emissions_unit_mapping['N2O'] = 'Mt N2O/yr'

# %%
emissions_unit_mapping

# %%
# others
# Heat Content|Ocean
# Heat Uptake
# Heat Uptake|Other
# Surface Air Temperature Change

# %%
physics_unit_mapping = {
    "Surface Air Temperature Change": 'K',
    "Heat Content|Ocean": 'ZJ',
    "Heat Uptake|Ocean": 'ZJ/yr',
    "Heat Uptake": 'ZJ/yr',
}

# %%
climate_model = "fair-2.2.4_cal-1.6.1"
region = "World"
model = "rcmip"

# %%
aggregated_forcers={
    "Natural": ["Solar", "Volcanic"],
    "Anthropogenic": [
        "CO2",
        "CH4",
        "N2O",
        "CF4",
        "C2F6",
        "C3F8",
        "c-C4F8",
        "C4F10",
        "C5F12",
        "C6F14",
        "C7F16",
        "C8F18",
        "NF3",
        "SF6",
        "SO2F2",
        "HFC-125",
        "HFC-134a",
        "HFC-143a",
        "HFC-152a",
        "HFC-227ea",
        "HFC-23",
        "HFC-236fa",
        "HFC-245fa",
        "HFC-32",
        "HFC-365mfc",
        "HFC-4310mee",
        "CFC-11",
        "CFC-12",
        "CFC-113",
        "CFC-114",
        "CFC-115",
        "HCFC-22",
        "HCFC-141b",
        "HCFC-142b",
        "CCl4",
        "CHCl3",
        "CH2Cl2",
        "CH3Cl",
        "CH3CCl3",
        "CH3Br",
        "Halon-1211",
        "Halon-1301",
        "Halon-2402",
        "Ozone",
        "Light absorbing particles on snow and ice", 
        "Contrails and Contrail-induced Cirrus", 
        "Stratospheric water vapour",
        "Aerosol-radiation interactions", 
        "Aerosol-cloud interactions",
        "Irrigation",
        "Land use",
        "Albedo Change",  # CMIP6 not aggregated
    ],
    "Aerosol": ["Aerosol-radiation interactions", "Aerosol-cloud interactions"],
    "Albedo Change": ["Irrigation", "Land use"],
    "Other": ["Light absorbing particles on snow and ice", "Contrails and Contrail-induced Cirrus", "Stratospheric water vapour"],
    "F-Gases": [
        "CF4",
        "C2F6",
        "C3F8",
        "c-C4F8",
        "C4F10",
        "C5F12",
        "C6F14",
        "C7F16",
        "C8F18",
        "NF3",
        "SF6",
        "SO2F2",
        "HFC-125",
        "HFC-134a",
        "HFC-143a",
        "HFC-152a",
        "HFC-227ea",
        "HFC-23",
        "HFC-236fa",
        "HFC-245fa",
        "HFC-32",
        "HFC-365mfc",
        "HFC-4310mee",
    ],
    "Montreal Gases": [
        "CFC-11",
        "CFC-12",
        "CFC-113",
        "CFC-114",
        "CFC-115",
        "HCFC-22",
        "HCFC-141b",
        "HCFC-142b",
        "CCl4",
        "CHCl3",
        "CH2Cl2",
        "CH3Cl",
        "CH3CCl3",
        "CH3Br",
        "Halon-1211",
        "Halon-1301",
        "Halon-2402",
    ],
    "HFC": [
        "HFC-125",
        "HFC-134a",
        "HFC-143a",
        "HFC-152a",
        "HFC-227ea",
        "HFC-23",
        "HFC-236fa",
        "HFC-245fa",
        "HFC-32",
        "HFC-365mfc",
        "HFC-4310mee",
    ],
    "CFC": [
        "CFC-11",
        "CFC-12",
        "CFC-113",
        "CFC-114",
        "CFC-115",
    ],
    "PFC": [
        "CF4",
        "C2F6",
        "C3F8",
        "c-C4F8",
        "C4F10",
        "C5F12",
        "C6F14",
        "C7F16",
        "C8F18",
    ],
}
aggregated_forcing_variable_mapping = {
    "Anthropogenic": 'Effective Radiative Forcing|Anthropogenic',
    "Aerosol": 'Effective Radiative Forcing|Anthropogenic|Aerosol',
    "Albedo Change": 'Effective Radiative Forcing|Anthropogenic|Albedo Change',
    "Other": 'Effective Radiative Forcing|Anthropogenic|Other',
    "F-Gases": 'Effective Radiative Forcing|Anthropogenic|F-Gases',
    "Montreal Gases": 'Effective Radiative Forcing|Anthropogenic|Montreal Gases',
    "Natural": 'Effective Radiative Forcing|Natural',
    "HFC": 'Effective Radiative Forcing|Anthropogenic|F-Gases|HFC',
    "CFC": 'Effective Radiative Forcing|Anthropogenic|Montreal Gases|CFC',
    "PFC": 'Effective Radiative Forcing|Anthropogenic|F-Gases|PFC',
}

# %%
# start with purely emissions driven run scenarios as these should be easy
# piControl
# esm-piControl
# esm-allGHG-piControl
# 1pctCO2
# 1pctCO2-4xext
# 1pctCO2-cdr
# 1pctCO2-bgc
# 1pctCO2-rad
# esm-1pct-brch-1000PgC
# esm-1pct-brch-2000PgC
# esm-1pct-brch-750PgC
# abrupt-4xCO2
# abrupt-2xCO2
# abrupt-0p5xCO2
# esm-pi-cdr-pulse
# esm-pi-CO2pulse
# esm-bell-1000PgC
# esm-bell-2000PgC
# esm-bell-750PgC
# historical
# historical-cmip6
# hist-aer
# hist-GHG
# hist-CO2
# ssp119
# ssp126
# ssp245
# ssp370
# ssp434
# ssp460
# ssp534-over
# ssp585
# esm-hist
# esm-hist-cmip6
# esm-ssp119
# esm-ssp126
# esm-ssp245
# esm-ssp370
# esm-ssp434
# esm-ssp460
# esm-ssp534-over
# esm-ssp585
# esm-allGHG-hist
# esm-allGHG-hist-cmip6
# esm-allGHG-ssp119
# esm-allGHG-ssp126
# esm-allGHG-ssp245
# esm-allGHG-ssp370
# esm-allGHG-ssp370-lowNTCF
# esm-allGHG-ssp370-lowCH4
# esm-allGHG-ssp370-lowNTCF-HighCH4
# esm-allGHG-ssp434
# esm-allGHG-ssp460
# esm-allGHG-ssp534-over
# esm-allGHG-ssp534-over-highCH4
# esm-allGHG-ssp585
# esm-allGHG-ssp585-lowCH4
# esm-scen7-H
# esm-scen7-HL
# esm-scen7-M
# esm-scen7-ML
# esm-scen7-L
# esm-scen7-VL
# esm-scen7-LN
# scen7-HC
# scen7-HLC
# scen7-MC
# scen7-MLC
# scen7-LC
# scen7-VLC
# scen7-LNC
# esm-allGHG-scen7-H
# esm-allGHG-scen7-HL
# esm-allGHG-scen7-H-CH4L
# esm-allGHG-scen7-M
# esm-allGHG-scen7-ML
# esm-allGHG-scen7-L
# esm-allGHG-scen7-L-CH4H
# esm-allGHG-scen7-VL
# esm-allGHG-scen7-LN
# esm-flat10
# esm-flat10-zec
# esm-flat10-cdr
# esm-flat10-nz
# esm-flat10-rev
# esm-flat7.5
# esm-flat7.5-cdr
# esm-flat7.5-zec
# esm-flat7.5-nz
# esm-flat7.5-rev
# esm-flat20
# esm-flat20-cdr
# esm-flat20-zec
# esm-flat20-nz
# esm-flat20-rev
# methanemip-TM-allGHG
# methanemip-TM+BC-allGHG


# %%
run_type = {
    "esm-allGHG-piControl": "idealised",
    "esm-allGHG-hist": "cmip7",
    "esm-allGHG-hist-cmip6": "cmip6",
    "esm-allGHG-ssp119": "cmip6",
    "esm-allGHG-ssp126": "cmip6",
    "esm-allGHG-ssp245": "cmip6",
    "esm-allGHG-ssp370": "cmip6",
    "esm-allGHG-ssp370-lowNTCF": "cmip6",
    "esm-allGHG-ssp370-lowCH4": "cmip6",
    "esm-allGHG-ssp370-lowNTCF-HighCH4": "cmip6",
    "esm-allGHG-ssp434": "cmip6",
    "esm-allGHG-ssp460": "cmip6",
    "esm-allGHG-ssp534-over": "cmip6",
    "esm-allGHG-ssp534-over-highCH4": "cmip6",
    "esm-allGHG-ssp585": "cmip6",
    "esm-allGHG-ssp585-lowCH4": "cmip6",
    "esm-allGHG-scen7-H": "cmip7",
    "esm-allGHG-scen7-HL": "cmip7",
    "esm-allGHG-scen7-H-CH4L": "cmip7",
    "esm-allGHG-scen7-M": "cmip7",
    "esm-allGHG-scen7-ML": "cmip7",
    "esm-allGHG-scen7-L": "cmip7",
    "esm-allGHG-scen7-L-CH4H": "cmip7",
    "esm-allGHG-scen7-VL": "cmip7",
    "esm-allGHG-scen7-LN": "cmip7",
    "methanemip-TM-allGHG": "cmip6",
    "methanemip-TM+BC-allGHG": "cmip6",
    "historical": "cmip7",
    "historical-cmip6": "cmip6",
    "ssp119": "cmip6",
    "ssp126": "cmip6",
    "ssp245": "cmip6",
    "ssp370": "cmip6",
    "ssp434": "cmip6",
    "ssp460": "cmip6",
    "ssp534-over": "cmip6",
    "ssp585": "cmip6",
    "esm-hist": "cmip7",
    "esm-hist-cmip6": "cmip6",
    "esm-ssp119": "cmip6",
    "esm-ssp126": "cmip6",
    "esm-ssp245": "cmip6",
    "esm-ssp370": "cmip6",
    "esm-ssp434": "cmip6",
    "esm-ssp460": "cmip6",
    "esm-ssp534-over": "cmip6",
    "esm-ssp585": "cmip6",
    "esm-scen7-H": "cmip7",
    "esm-scen7-HL": "cmip7",
    "esm-scen7-M": "cmip7",
    "esm-scen7-ML": "cmip7",
    "esm-scen7-L": "cmip7",
    "esm-scen7-VL": "cmip7",
    "esm-scen7-LN": "cmip7",
    "scen7-HC": "cmip7",
    "scen7-HLC": "cmip7",
    "scen7-MC": "cmip7",
    "scen7-MLC": "cmip7",
    "scen7-LC": "cmip7",
    "scen7-VLC": "cmip7",
    "scen7-LNC": "cmip7",
}

# %%
forcing_excludes = {
    'idealised': [
        "C3F8",
        "c-C4F8",
        "C4F10",
        "C5F12",
        "C7F16",
        "C8F18",
        "NF3",
        "SF6",
        "SO2F2",
        "HFC-152a",
        "HFC-236fa",
        "HFC-32",
        "HFC-365mfc",
        "CHCl3",
        "CH2Cl2",
        'Contrails and Contrail-induced Cirrus',
        'Albedo Change',
        'Irrigation',
        'Land use',
        'Solar',
        'Volcanic'
    ],
    'cmip7': ['Contrails and Contrail-induced Cirrus', 'Albedo Change'],
    'cmip6': ['Irrigation', 'Land use'],
    'hist-GHG': [
        'Aerosol-radiation interactions',
        'Aerosol-cloud interactions',
        'Ozone',
        'Light absorbing particles on snow and ice',
        'Stratospheric water vapour',
        'Contrails and Contrail-induced Cirrus',
        'Albedo Change',
        'Irrigation',
        'Land use',
    ]
}

category_excludes = {
    'idealised': ["Albedo Change", "Natural"],
    'cmip7': [],
    'cmip6': ['Albedo Change']
}

# %%
aggregated_forcers

# %%
for scenario in [
    "esm-allGHG-piControl",
    "esm-allGHG-hist",
    "esm-allGHG-hist-cmip6",
    "esm-allGHG-ssp119",
    "esm-allGHG-ssp126",
    "esm-allGHG-ssp245",
    "esm-allGHG-ssp370",
    "esm-allGHG-ssp370-lowNTCF",
    "esm-allGHG-ssp370-lowCH4",
    "esm-allGHG-ssp370-lowNTCF-HighCH4",
    "esm-allGHG-ssp434",
    "esm-allGHG-ssp460",
    "esm-allGHG-ssp534-over",
    "esm-allGHG-ssp534-over-highCH4",
    "esm-allGHG-ssp585",
    "esm-allGHG-ssp585-lowCH4",
    "esm-allGHG-scen7-H",
    "esm-allGHG-scen7-HL",
    "esm-allGHG-scen7-H-CH4L",
    "esm-allGHG-scen7-M",
    "esm-allGHG-scen7-ML",
    "esm-allGHG-scen7-L",
    "esm-allGHG-scen7-L-CH4H",
    "esm-allGHG-scen7-VL",
    "esm-allGHG-scen7-LN",
    "methanemip-TM-allGHG",
    "methanemip-TM+BC-allGHG",
]:
    ds_main = xr.load_dataset(f'../output/native/{scenario}.nc')
    ds_life = xr.load_dataset(f'../output/native/{scenario}_lifetimes.nc')

    physics_mapping = {
        "Surface Air Temperature Change": ds_main.temperature.sel(layer=0, scenario=scenario),
        "Heat Content|Ocean": ds_main.ocean_heat_content_change.sel(scenario=scenario) * 0.91 / 1e21,
        "Heat Uptake|Ocean": ds_main.toa_imbalance.sel(scenario=scenario) * earth_radius**2 * 4 * np.pi * seconds_per_year / 1e21 * 0.91,
        "Heat Uptake": ds_main.toa_imbalance.sel(scenario=scenario) * earth_radius**2 * 4 * np.pi * seconds_per_year / 1e21,
    }
    
    multi_dimensional_run = []
    
    # physical quantities
    for variable in physics_mapping:
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=physics_mapping[variable],
                    index=ds_main.timebound,
                    columns={
                        "variable": variable,
                        "unit": physics_unit_mapping[variable],
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # lifetimes
    for variable in lifetime_variable_mapping:
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_life["__xarray_dataarray_variable__"].sel(specie=variable),
                    index=ds_life.timebounds,
                    columns={
                        "variable": lifetime_variable_mapping[variable],
                        "unit": 'yr',
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_life.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # concentrations
    for variable in concentrations_variable_mapping:
        if variable in forcing_excludes[run_type[scenario]]:
            continue
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_main.concentration.sel(scenario=scenario, specie=variable),
                    index=ds_main.timebound,
                    columns={
                        "variable": concentrations_variable_mapping[variable],
                        "unit": concentrations_unit_mapping[variable],
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # single forcing categories
    for variable in forcing_variable_mapping:
        if variable in forcing_excludes[run_type[scenario]]:
            continue
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_main.forcing.sel(scenario=scenario, specie=variable),
                    index=ds_main.timebound,
                    columns={
                        "variable": forcing_variable_mapping[variable],
                        "unit": "W/m^2",
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # total forcing
    multi_dimensional_run.extend(
        [
            ScmRun(
                data=ds_main.forcing_sum.sel(scenario=scenario),
                index=ds_main.timebound,
                columns={
                    "variable": "Effective Radiative Forcing",
                    "unit": "W/m^2",
                    "region": region,
                    "model": model,
                    "scenario": scenario,
                    "ensemble_member": ds_main.config,
                    "climate_model": climate_model
                },
            )
        ]
    )
    
    # aggregated forcing categories
    for category in aggregated_forcers:
        if category in category_excludes[run_type[scenario]]:
            continue
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=sum(
                        [
                            ds_main.forcing.sel(scenario=scenario, specie=variable) 
                            for variable in aggregated_forcers[category] if variable not in forcing_excludes[run_type[scenario]]
                        ]
                    ),
                    index=ds_main.timebound,
                    columns={
                        "variable": aggregated_forcing_variable_mapping[category],
                        "unit": "W/m^2",
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    multi_dimensional_run = run_append(multi_dimensional_run)
    
    multi_dimensional_run.to_nc(f"../output/pyrcmip/{scenario}.nc", dimensions=["ensemble_member"])

# %%
# hist-aer a special case

scenario = 'hist-aer'

ds_main = xr.load_dataset(f'../output/native/{scenario}.nc')

physics_mapping = {
    "Surface Air Temperature Change": ds_main.temperature.sel(layer=0, scenario=scenario),
    "Heat Content|Ocean": ds_main.ocean_heat_content_change.sel(scenario=scenario) * 0.91 / 1e21,
    "Heat Uptake|Ocean": ds_main.toa_imbalance.sel(scenario=scenario) * earth_radius**2 * 4 * np.pi * seconds_per_year / 1e21 * 0.91,
    "Heat Uptake": ds_main.toa_imbalance.sel(scenario=scenario) * earth_radius**2 * 4 * np.pi * seconds_per_year / 1e21,
}

multi_dimensional_run = []

# physical quantities
for variable in physics_mapping:
    multi_dimensional_run.extend(
        [
            ScmRun(
                data=physics_mapping[variable],
                index=ds_main.timebound,
                columns={
                    "variable": variable,
                    "unit": physics_unit_mapping[variable],
                    "region": region,
                    "model": model,
                    "scenario": scenario,
                    "ensemble_member": ds_main.config,
                    "climate_model": climate_model
                },
            )
        ]
    )

# single forcing categories
for variable in ["Aerosol-radiation interactions", "Aerosol-cloud interactions"]:
    multi_dimensional_run.extend(
        [
            ScmRun(
                data=ds_main.forcing.sel(scenario=scenario, specie=variable),
                index=ds_main.timebound,
                columns={
                    "variable": forcing_variable_mapping[variable],
                    "unit": "W/m^2",
                    "region": region,
                    "model": model,
                    "scenario": scenario,
                    "ensemble_member": ds_main.config,
                    "climate_model": climate_model
                },
            )
        ]
    )

# total forcing
multi_dimensional_run.extend(
    [
        ScmRun(
            data=ds_main.forcing_sum.sel(scenario=scenario),
            index=ds_main.timebound,
            columns={
                "variable": "Effective Radiative Forcing",
                "unit": "W/m^2",
                "region": region,
                "model": model,
                "scenario": scenario,
                "ensemble_member": ds_main.config,
                "climate_model": climate_model
            },
        )
    ]
)

# aggregated forcing categories
for category in ["Aerosol"]:
    multi_dimensional_run.extend(
        [
            ScmRun(
                data=sum(
                    [
                        ds_main.forcing.sel(scenario=scenario, specie=variable) 
                        for variable in aggregated_forcers[category]
                    ]
                ),
                index=ds_main.timebound,
                columns={
                    "variable": aggregated_forcing_variable_mapping[category],
                    "unit": "W/m^2",
                    "region": region,
                    "model": model,
                    "scenario": scenario,
                    "ensemble_member": ds_main.config,
                    "climate_model": climate_model
                },
            )
        ]
    )

multi_dimensional_run = run_append(multi_dimensional_run)

multi_dimensional_run.to_nc(f"../output/pyrcmip/{scenario}.nc", dimensions=["ensemble_member"])

# %%
# aggregated_forcers={
#     "Anthropogenic": ["CO2"]
# }

# %%
# now we do the ones that are only CO2-concentration-driven

for scenario in [
    "piControl",
    "1pctCO2",
    "1pctCO2-4xext",
    "1pctCO2-cdr",
    "1pctCO2-bgc",
    "1pctCO2-rad",
    "abrupt-4xCO2",
    "abrupt-2xCO2",
    "abrupt-0p5xCO2",
    "hist-CO2"
]:
    
    ds_main = xr.load_dataset(f'../output/native/{scenario}.nc')

    physics_mapping = {
        "Surface Air Temperature Change": ds_main.temperature.sel(layer=0, scenario=scenario),
        "Heat Content|Ocean": ds_main.ocean_heat_content_change.sel(scenario=scenario) * 0.91 / 1e21,
        "Heat Uptake|Ocean": ds_main.toa_imbalance.sel(scenario=scenario) * earth_radius**2 * 4 * np.pi * seconds_per_year / 1e21 * 0.91,
        "Heat Uptake": ds_main.toa_imbalance.sel(scenario=scenario) * earth_radius**2 * 4 * np.pi * seconds_per_year / 1e21,
    }
    
    multi_dimensional_run = []
    
    # physical quantities
    for variable in physics_mapping:
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=physics_mapping[variable],
                    index=ds_main.timebound,
                    columns={
                        "variable": variable,
                        "unit": physics_unit_mapping[variable],
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # emissions
    for variable in ["CO2"]:
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_main.emissions.sel(scenario=scenario, specie=variable),
                    index=ds_main.timepoint,
                    columns={
                        "variable": emissions_variable_mapping[variable],
                        "unit": emissions_unit_mapping[variable],
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # single forcing categories
    for variable in ["CO2"]:
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_main.forcing.sel(scenario=scenario, specie=variable),
                    index=ds_main.timebound,
                    columns={
                        "variable": forcing_variable_mapping[variable],
                        "unit": "W/m^2",
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # total forcing
    multi_dimensional_run.extend(
        [
            ScmRun(
                data=ds_main.forcing_sum.sel(scenario=scenario),
                index=ds_main.timebound,
                columns={
                    "variable": "Effective Radiative Forcing",
                    "unit": "W/m^2",
                    "region": region,
                    "model": model,
                    "scenario": scenario,
                    "ensemble_member": ds_main.config,
                    "climate_model": climate_model
                },
            )
        ]
    )
    
    # aggregated forcing categories
    for category in ["Anthropogenic"]:
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_main.forcing.sel(scenario=scenario, specie="CO2"),
                    index=ds_main.timebound,
                    columns={
                        "variable": aggregated_forcing_variable_mapping[category],
                        "unit": "W/m^2",
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    multi_dimensional_run = run_append(multi_dimensional_run)
    
    multi_dimensional_run.to_nc(f"../output/pyrcmip/{scenario}.nc", dimensions=["ensemble_member"])

# %%
# hist-GHG is a special case being only C-driven for more than CO2

scenario = "hist-GHG"

ds_main = xr.load_dataset(f'../output/native/{scenario}.nc')

physics_mapping = {
    "Surface Air Temperature Change": ds_main.temperature.sel(layer=0, scenario=scenario),
    "Heat Content|Ocean": ds_main.ocean_heat_content_change.sel(scenario=scenario) * 0.91 / 1e21,
    "Heat Uptake|Ocean": ds_main.toa_imbalance.sel(scenario=scenario) * earth_radius**2 * 4 * np.pi * seconds_per_year / 1e21 * 0.91,
    "Heat Uptake": ds_main.toa_imbalance.sel(scenario=scenario) * earth_radius**2 * 4 * np.pi * seconds_per_year / 1e21,
}

multi_dimensional_run = []

# physical quantities
for variable in physics_mapping:
    multi_dimensional_run.extend(
        [
            ScmRun(
                data=physics_mapping[variable],
                index=ds_main.timebound,
                columns={
                    "variable": variable,
                    "unit": physics_unit_mapping[variable],
                    "region": region,
                    "model": model,
                    "scenario": scenario,
                    "ensemble_member": ds_main.config,
                    "climate_model": climate_model
                },
            )
        ]
    )

# emissions
for variable in ["CO2", "CH4", "N2O"] + aggregated_forcers["F-Gases"] + aggregated_forcers["Montreal Gases"]:
    multi_dimensional_run.extend(
        [
            ScmRun(
                data=ds_main.emissions.sel(scenario=scenario, specie=variable),
                index=ds_main.timepoint,
                columns={
                    "variable": emissions_variable_mapping[variable],
                    "unit": emissions_unit_mapping[variable],
                    "region": region,
                    "model": model,
                    "scenario": scenario,
                    "ensemble_member": ds_main.config,
                    "climate_model": climate_model
                },
            )
        ]
    )

# single forcing categories
for variable in ["CO2", "CH4", "N2O"] + aggregated_forcers["F-Gases"] + aggregated_forcers["Montreal Gases"]:
    multi_dimensional_run.extend(
        [
            ScmRun(
                data=ds_main.forcing.sel(scenario=scenario, specie=variable),
                index=ds_main.timebound,
                columns={
                    "variable": forcing_variable_mapping[variable],
                    "unit": "W/m^2",
                    "region": region,
                    "model": model,
                    "scenario": scenario,
                    "ensemble_member": ds_main.config,
                    "climate_model": climate_model
                },
            )
        ]
    )

# total forcing
multi_dimensional_run.extend(
    [
        ScmRun(
            data=ds_main.forcing_sum.sel(scenario=scenario),
            index=ds_main.timebound,
            columns={
                "variable": "Effective Radiative Forcing",
                "unit": "W/m^2",
                "region": region,
                "model": model,
                "scenario": scenario,
                "ensemble_member": ds_main.config,
                "climate_model": climate_model
            },
        )
    ]
)

# aggregated forcing categories
for category in ["Anthropogenic", "F-Gases", "Montreal Gases", "HFC", "PFC", "CFC"]:
    multi_dimensional_run.extend(
        [
            ScmRun(
                data=sum(
                    [
                        ds_main.forcing.sel(scenario=scenario, specie=variable) 
                        for variable in aggregated_forcers[category] if variable not in forcing_excludes['hist-GHG']
                    ]
                ),
                index=ds_main.timebound,
                columns={
                    "variable": aggregated_forcing_variable_mapping[category],
                    "unit": "W/m^2",
                    "region": region,
                    "model": model,
                    "scenario": scenario,
                    "ensemble_member": ds_main.config,
                    "climate_model": climate_model
                },
            )
        ]
    )

multi_dimensional_run = run_append(multi_dimensional_run)

multi_dimensional_run.to_nc(f"../output/pyrcmip/{scenario}.nc", dimensions=["ensemble_member"])

# %%
# Now do emissions-driven but where we only care about CO2

for scenario in [
    "esm-piControl",
    "esm-1pct-brch-1000PgC",
    "esm-1pct-brch-2000PgC",
    "esm-1pct-brch-750PgC",
    "esm-pi-cdr-pulse",
    "esm-pi-CO2pulse",
    "esm-bell-1000PgC",
    "esm-bell-2000PgC",
    "esm-bell-750PgC",
    "esm-flat10",
    "esm-flat10-zec",
    "esm-flat10-cdr",
    "esm-flat10-nz",
    "esm-flat10-rev",
    "esm-flat7.5",
    "esm-flat7.5-cdr",
    "esm-flat7.5-zec",
    "esm-flat7.5-nz",
    "esm-flat7.5-rev",
    "esm-flat20",
    "esm-flat20-cdr",
    "esm-flat20-zec",
    "esm-flat20-nz",
    "esm-flat20-rev",
]:
    ds_main = xr.load_dataset(f'../output/native/{scenario}.nc')
    
    physics_mapping = {
        "Surface Air Temperature Change": ds_main.temperature.sel(layer=0, scenario=scenario),
        "Heat Content|Ocean": ds_main.ocean_heat_content_change.sel(scenario=scenario) * 0.91 / 1e21,
        "Heat Uptake|Ocean": ds_main.toa_imbalance.sel(scenario=scenario) * earth_radius**2 * 4 * np.pi * seconds_per_year / 1e21 * 0.91,
        "Heat Uptake": ds_main.toa_imbalance.sel(scenario=scenario) * earth_radius**2 * 4 * np.pi * seconds_per_year / 1e21,
    }
    
    multi_dimensional_run = []
    
    # physical quantities
    for variable in physics_mapping:
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=physics_mapping[variable],
                    index=ds_main.timebound,
                    columns={
                        "variable": variable,
                        "unit": physics_unit_mapping[variable],
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # concentrations
    for variable in ["CO2"]:
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_main.concentration.sel(scenario=scenario, specie=variable),
                    index=ds_main.timebound,
                    columns={
                        "variable": concentrations_variable_mapping[variable],
                        "unit": concentrations_unit_mapping[variable],
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # single forcing categories
    for variable in ["CO2"]:
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_main.forcing.sel(scenario=scenario, specie=variable),
                    index=ds_main.timebound,
                    columns={
                        "variable": forcing_variable_mapping[variable],
                        "unit": "W/m^2",
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # total forcing
    multi_dimensional_run.extend(
        [
            ScmRun(
                data=ds_main.forcing_sum.sel(scenario=scenario),
                index=ds_main.timebound,
                columns={
                    "variable": "Effective Radiative Forcing",
                    "unit": "W/m^2",
                    "region": region,
                    "model": model,
                    "scenario": scenario,
                    "ensemble_member": ds_main.config,
                    "climate_model": climate_model
                },
            )
        ]
    )
    
    # aggregated forcing categories
    for category in ["Anthropogenic"]:
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_main.forcing.sel(scenario=scenario, specie="CO2"),
                    index=ds_main.timebound,
                    columns={
                        "variable": aggregated_forcing_variable_mapping[category],
                        "unit": "W/m^2",
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    multi_dimensional_run = run_append(multi_dimensional_run)
    
    multi_dimensional_run.to_nc(f"../output/pyrcmip/{scenario}.nc", dimensions=["ensemble_member"])

# %%
# finally scenarios with a mix of both

for scenario in [
    "historical",
    "historical-cmip6",
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp434",
    "ssp460",
    "ssp534-over",
    "ssp585",
    "esm-hist",
    "esm-hist-cmip6",
    "esm-ssp119",
    "esm-ssp126",
    "esm-ssp245",
    "esm-ssp370",
    "esm-ssp434",
    "esm-ssp460",
    "esm-ssp534-over",
    "esm-ssp585",
    "esm-scen7-H",
    "esm-scen7-HL",
    "esm-scen7-M",
    "esm-scen7-ML",
    "esm-scen7-L",
    "esm-scen7-VL",
    "esm-scen7-LN",
    "scen7-HC",
    "scen7-HLC",
    "scen7-MC",
    "scen7-MLC",
    "scen7-LC",
    "scen7-VLC",
    "scen7-LNC",
]:
    ds_main = xr.load_dataset(f'../output/native/{scenario}.nc')

    physics_mapping = {
        "Surface Air Temperature Change": ds_main.temperature.sel(layer=0, scenario=scenario),
        "Heat Content|Ocean": ds_main.ocean_heat_content_change.sel(scenario=scenario) * 0.91 / 1e21,
        "Heat Uptake|Ocean": ds_main.toa_imbalance.sel(scenario=scenario) * earth_radius**2 * 4 * np.pi * seconds_per_year / 1e21 * 0.91,
        "Heat Uptake": ds_main.toa_imbalance.sel(scenario=scenario) * earth_radius**2 * 4 * np.pi * seconds_per_year / 1e21,
    }
    
    multi_dimensional_run = []
    
    # physical quantities
    for variable in physics_mapping:
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=physics_mapping[variable],
                    index=ds_main.timebound,
                    columns={
                        "variable": variable,
                        "unit": physics_unit_mapping[variable],
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # lifetimes
    ds_life = xr.load_dataset(f'../output/native/{scenario}_lifetimes.nc')
    for variable in lifetime_variable_mapping:
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_life["__xarray_dataarray_variable__"].sel(specie=variable),
                    index=ds_life.timebounds,
                    columns={
                        "variable": lifetime_variable_mapping[variable],
                        "unit": 'yr',
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_life.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # concentrations
    for variable in concentrations_variable_mapping:
        if variable in forcing_excludes[run_type[scenario]]:
            continue
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_main.concentration.sel(scenario=scenario, specie=variable),
                    index=ds_main.timebound,
                    columns={
                        "variable": concentrations_variable_mapping[variable],
                        "unit": concentrations_unit_mapping[variable],
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )

    # emissions
    for variable in ["CO2", "CH4", "N2O"] + aggregated_forcers["F-Gases"] + aggregated_forcers["Montreal Gases"]:
        if variable in forcing_excludes[run_type[scenario]]:
            continue
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_main.emissions.sel(scenario=scenario, specie=variable),
                    index=ds_main.timepoint,
                    columns={
                        "variable": emissions_variable_mapping[variable],
                        "unit": emissions_unit_mapping[variable],
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # single forcing categories
    for variable in forcing_variable_mapping:
        if variable in forcing_excludes[run_type[scenario]]:
            continue
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=ds_main.forcing.sel(scenario=scenario, specie=variable),
                    index=ds_main.timebound,
                    columns={
                        "variable": forcing_variable_mapping[variable],
                        "unit": "W/m^2",
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    # total forcing
    multi_dimensional_run.extend(
        [
            ScmRun(
                data=ds_main.forcing_sum.sel(scenario=scenario),
                index=ds_main.timebound,
                columns={
                    "variable": "Effective Radiative Forcing",
                    "unit": "W/m^2",
                    "region": region,
                    "model": model,
                    "scenario": scenario,
                    "ensemble_member": ds_main.config,
                    "climate_model": climate_model
                },
            )
        ]
    )
    
    # aggregated forcing categories
    for category in aggregated_forcers:
        if category in category_excludes[run_type[scenario]]:
            continue
        multi_dimensional_run.extend(
            [
                ScmRun(
                    data=sum(
                        [
                            ds_main.forcing.sel(scenario=scenario, specie=variable) 
                            for variable in aggregated_forcers[category] if variable not in forcing_excludes[run_type[scenario]]
                        ]
                    ),
                    index=ds_main.timebound,
                    columns={
                        "variable": aggregated_forcing_variable_mapping[category],
                        "unit": "W/m^2",
                        "region": region,
                        "model": model,
                        "scenario": scenario,
                        "ensemble_member": ds_main.config,
                        "climate_model": climate_model
                    },
                )
            ]
        )
    
    multi_dimensional_run = run_append(multi_dimensional_run)
    
    multi_dimensional_run.to_nc(f"../output/pyrcmip/{scenario}.nc", dimensions=["ensemble_member"])

# %%
