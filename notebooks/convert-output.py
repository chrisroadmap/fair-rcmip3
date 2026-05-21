# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Convert output from fair to rcmip format

# %%
from fair.earth_params import molecular_weight_air, mass_atmosphere
import pandas as pd
import xarray as xr
from scmdata.run import ScmRun, run_append
from scmdata.netcdf import nc_to_run

# %%
CO2_MASS = 44.009
C_MASS = 12.011


# %%
def new_timeseries(
    n=100,
    count=1,
    model="example",
    scenario="ssp119",
    variable="Surface Temperature",
    unit="K",
    region="World",
    cls=ScmRun,
    **kwargs,
):
    data = np.random.rand(n, count) * np.arange(n)[:, np.newaxis]
    index = 1750 + np.arange(n)
    return cls(
        data,
        columns={
            "model": model,
            "scenario": scenario,
            "variable": variable,
            "region": region,
            "unit": unit,
            **kwargs,
        },
        index=index,
    )


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
variable_mapping

# %%
concentrations_variable_mapping = {variable_mapping[var.split('|')[-1]]:var for var in output_variables if var.startswith('Atmospheric Concentrations')}

# %%
concentrations_variable_mapping

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
forcing_variable_mapping["HFC"] = forcing_variable_mapping.pop("HFC-")
forcing_variable_mapping["CFC"] = forcing_variable_mapping.pop("CFC-")
forcing_variable_mapping

# %%
output_variables

# %%
carbonpool_variable_mapping = {"CO2": "Carbon Pool|Atmosphere"}

# %%

Carbon Pool|Atmosphere
Carbon Pool|Land
Carbon Pool|Land|Litter
Carbon Pool|Land|Soil
Carbon Pool|Land|Vegetation
Carbon Pool|Land|Wood Products
Carbon Pool|Ocean
Carbon Pool|Ocean|Deep
Carbon Pool|Ocean|Deep|Inorganic
Carbon Pool|Ocean|Deep|Organic
Carbon Pool|Ocean|Surface
Carbon Pool|Ocean|Surface|Inorganic
Carbon Pool|Ocean|Surface|Organic
Carbon Sequestration
Effective Radiative Forcing
Effective Radiative Forcing|Anthropogenic
Effective Radiative Forcing|Anthropogenic|Aerosol
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-cloud Interactions
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|BC
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Biomass Burning
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Biomass Burning|BC
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Biomass Burning|NH3
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Biomass Burning|Nitrate
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Biomass Burning|OC
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Biomass Burning|Sulfate
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Fossil and Industrial
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Fossil and Industrial|BC
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Fossil and Industrial|NH3
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Fossil and Industrial|Nitrate
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Fossil and Industrial|OC
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Fossil and Industrial|Sulfate
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Mineral Dust
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|NH3
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|NH3|Biomass Burning
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|NH3|Fossil and Industrial
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Nitrate
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Nitrate|Biomass Burning
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Nitrate|Fossil and Industrial
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|OC
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Other
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Sulfate
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Sulfate|Biomass Burning
Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions|Sulfate|Fossil and Industrial
Effective Radiative Forcing|Anthropogenic|Albedo Change
Effective Radiative Forcing|Anthropogenic|Albedo Change|Irrigation
Effective Radiative Forcing|Anthropogenic|Albedo Change|Land use
Effective Radiative Forcing|Anthropogenic|CH4
Effective Radiative Forcing|Anthropogenic|CO2
Effective Radiative Forcing|Anthropogenic|F-Gases
Effective Radiative Forcing|Anthropogenic|F-Gases|HFC
Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC125
Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC134a
Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC143a
Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC152a
Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC227ea
Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC23
Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC236fa
Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC245fa
Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC32
Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC365mfc
Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC4310mee
Effective Radiative Forcing|Anthropogenic|F-Gases|NF3
Effective Radiative Forcing|Anthropogenic|F-Gases|PFC
Effective Radiative Forcing|Anthropogenic|F-Gases|PFC|C2F6
Effective Radiative Forcing|Anthropogenic|F-Gases|PFC|C3F8
Effective Radiative Forcing|Anthropogenic|F-Gases|PFC|C4F10
Effective Radiative Forcing|Anthropogenic|F-Gases|PFC|C5F12
Effective Radiative Forcing|Anthropogenic|F-Gases|PFC|C6F14
Effective Radiative Forcing|Anthropogenic|F-Gases|PFC|C7F16
Effective Radiative Forcing|Anthropogenic|F-Gases|PFC|C8F18
Effective Radiative Forcing|Anthropogenic|F-Gases|PFC|cC4F8
Effective Radiative Forcing|Anthropogenic|F-Gases|PFC|CF4
Effective Radiative Forcing|Anthropogenic|F-Gases|SF6
Effective Radiative Forcing|Anthropogenic|F-Gases|SO2F2
Effective Radiative Forcing|Anthropogenic|Montreal Gases
Effective Radiative Forcing|Anthropogenic|Montreal Gases|CCl4
Effective Radiative Forcing|Anthropogenic|Montreal Gases|CFC
Effective Radiative Forcing|Anthropogenic|Montreal Gases|CFC|CFC11
Effective Radiative Forcing|Anthropogenic|Montreal Gases|CFC|CFC113
Effective Radiative Forcing|Anthropogenic|Montreal Gases|CFC|CFC114
Effective Radiative Forcing|Anthropogenic|Montreal Gases|CFC|CFC115
Effective Radiative Forcing|Anthropogenic|Montreal Gases|CFC|CFC12
Effective Radiative Forcing|Anthropogenic|Montreal Gases|CH2Cl2
Effective Radiative Forcing|Anthropogenic|Montreal Gases|CH3Br
Effective Radiative Forcing|Anthropogenic|Montreal Gases|CH3CCl3
Effective Radiative Forcing|Anthropogenic|Montreal Gases|CH3Cl
Effective Radiative Forcing|Anthropogenic|Montreal Gases|CHCl3
Effective Radiative Forcing|Anthropogenic|Montreal Gases|Halon1202
Effective Radiative Forcing|Anthropogenic|Montreal Gases|Halon1211
Effective Radiative Forcing|Anthropogenic|Montreal Gases|Halon1301
Effective Radiative Forcing|Anthropogenic|Montreal Gases|Halon2402
Effective Radiative Forcing|Anthropogenic|Montreal Gases|HCFC141b
Effective Radiative Forcing|Anthropogenic|Montreal Gases|HCFC142b
Effective Radiative Forcing|Anthropogenic|Montreal Gases|HCFC22
Effective Radiative Forcing|Anthropogenic|N2O
Effective Radiative Forcing|Anthropogenic|Other
Effective Radiative Forcing|Anthropogenic|Other|BC on Snow
Effective Radiative Forcing|Anthropogenic|Other|Contrails and Contrail-induced Cirrus
Effective Radiative Forcing|Anthropogenic|Other|Other WMGHGs
Effective Radiative Forcing|Anthropogenic|Other|Stratospheric H2O
Effective Radiative Forcing|Anthropogenic|Ozone
Effective Radiative Forcing|Anthropogenic|Ozone|Stratospheric Contribution
Effective Radiative Forcing|Anthropogenic|Ozone|Tropospheric Contribution
Effective Radiative Forcing|Natural
Effective Radiative Forcing|Natural|Solar
Effective Radiative Forcing|Natural|Volcanic
Emissions|BC
Net Flux to Atmosphere|CH4
Emissions|CH4
Emissions|CH4|Biomass Burning
Emissions|CH4|Fossil, Industry and AFOLU
Emissions|CH4|Other
Natural Fluxes|CH4
Natural Fluxes|CH4|Emissions
Natural Fluxes|CH4|Emissions|Other
Natural Fluxes|CH4|Emissions|Permafrost
Natural Fluxes|CH4|Emissions|Wetland
Natural Fluxes|CH4|Other
Natural Fluxes|CH4|Sink
Natural Fluxes|CH4|Sink|Other
Natural Fluxes|CH4|Sink|Soil
Natural Fluxes|CH4|Sink|Stratosphere
Natural Fluxes|CH4|Sink|Troposphere
Emissions|CO
Net Flux to Atmosphere|CO2
Emissions|CO2
Emissions|CO2|Fossil and Industry
Emissions|CO2|Land Use Change
Emissions|CO2|Other
Natural Fluxes|CO2
Natural Fluxes|CO2|Land                     
Natural Fluxes|CO2|Land|Permafrost
Natural Fluxes|CO2|Ocean
Natural Fluxes|CO2|Other
Emissions|F-Gases|HFC|HFC125
Emissions|F-Gases|HFC|HFC134a
Emissions|F-Gases|HFC|HFC143a
Emissions|F-Gases|HFC|HFC152a
Emissions|F-Gases|HFC|HFC227ea
Emissions|F-Gases|HFC|HFC23
Emissions|F-Gases|HFC|HFC236fa
Emissions|F-Gases|HFC|HFC245fa
Emissions|F-Gases|HFC|HFC32
Emissions|F-Gases|HFC|HFC365mfc
Emissions|F-Gases|HFC|HFC4310mee
Emissions|F-Gases|NF3
Emissions|F-Gases|PFC|C2F6
Emissions|F-Gases|PFC|C3F8
Emissions|F-Gases|PFC|C4F10
Emissions|F-Gases|PFC|C5F12
Emissions|F-Gases|PFC|C6F14
Emissions|F-Gases|PFC|C7F16
Emissions|F-Gases|PFC|C8F18
Emissions|F-Gases|PFC|cC4F8
Emissions|F-Gases|PFC|CF4
Emissions|F-Gases|SF6
Emissions|F-Gases|SO2F2
Emissions|Montreal Gases|CCl4
Emissions|Montreal Gases|CFC|CFC11
Emissions|Montreal Gases|CFC|CFC113
Emissions|Montreal Gases|CFC|CFC114
Emissions|Montreal Gases|CFC|CFC115
Emissions|Montreal Gases|CFC|CFC12
Emissions|Montreal Gases|CH2Cl2
Emissions|Montreal Gases|CH3Br
Emissions|Montreal Gases|CH3CCl3
Emissions|Montreal Gases|CH3Cl
Emissions|Montreal Gases|CHCl3
Emissions|Montreal Gases|Halon1202
Emissions|Montreal Gases|Halon1211
Emissions|Montreal Gases|Halon1301
Emissions|Montreal Gases|Halon2402
Emissions|Montreal Gases|HCFC141b
Emissions|Montreal Gases|HCFC142b
Emissions|Montreal Gases|HCFC22
Net Flux to Atmosphere|N2O
Emissions|N2O
Natural Fluxes|N2O
Emissions|NH3
Emissions|NOx
Emissions|OC
Emissions|Sulfur
Emissions|VOC
Heat Content|Ocean
Heat Content|Ocean|0-700m
Heat Content|Ocean|700-2000m
Heat Content|Ocean|below-2000m
Heat Uptake
Heat Uptake|Atmosphere
Heat Uptake|Ice
Heat Uptake|Land
Heat Uptake|Ocean
Heat Uptake|Other
Ocean pH
Sea Level Change
Sea Level Change|Thermal Expansion
Sea Level Change|Glaciers
Sea Level Change|Greenland
Sea Level Change|Antarctica
Sea Level Change|Land Water Storage
Surface Air Ocean Blended Temperature Change
Surface Air Temperature Change
Surface Ocean Temperature Change
    
}



# %%
ds = xr.load_dataset('../output/native/esm-allGHG-scen7-H-CH4L.nc')

# %%
ds

# %%
ds.concentration.sel(specie="CO2", scenario="esm-allGHG-scen7-H-CH4L") 

# %%
# ppm to GtCO2 conversion
mass_per_concentration = (
    mass_atmosphere
    / 1e18
    * CO2_MASS
    / molecular_weight_air
)

# %%
mass_per_concentration

# %%
ds.config

# %%
runs = run_append(
    [
        new_timeseries(
            results[:,NVAR*ensemble_member:NVAR*ensemble_member+NVAR],
            n=results.shape[0],
            scenario="piControl",
            variable=VARIABLES,
            unit=UNITS,
            ensemble_member=ensemble_member,
        )
        for ensemble_member in range(len(valid))
    ]
)
runs.metadata["source"] = "FaIR 1.6 RCMIP runs 27.08.2020"
runs.to_nc('../output/FaIR1.6_%s.nc' % expt, dimensions=["ensemble_member"])
