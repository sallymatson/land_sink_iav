import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def take_weighted_mean(da):
    '''
    Weighting based on tutorial from here:
    http://xarray.pydata.org/en/stable/examples/area_weighted_temperature.html
    Return weighted (by area) mean of a DataArray breaking it into N exatropics,
    tropics, and S exatropics.
    '''
    weights = np.cos(np.deg2rad(da.lat))  # Area is proportional to cosine of latitude
    weights.name = "weights"
    weighted_da = da.weighted(weights)  # xr is so slick
    da_latbands = weighted_da.mean(['lon'])
    weighted_means = da_latbands.groupby_bins(
        "lat", [-90, -30, 30, 90], labels=["South", "Tropics", "North"]
        ).mean(['lat'])
    return weighted_means


def open_wind():
    wind_filename = "weather/wind/wspd.sig995.mon.mean.nc"
    ds = xr.open_dataset(wind_filename)
    da = ds.wspd
    weighted_means = take_weighted_mean(da)
    df = weighted_means.to_pandas().add_prefix("WSPD_")
    df.index = df.index.to_period("M")
    return df


def open_landtemp_netcdf():
    land_filename = "weather/temp/CRUTEM.5.0.1.0.alt.anomalies.nc"
    ds = xr.open_dataset(land_filename)
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    da = ds.tas
    weighted_means = take_weighted_mean(da)
    df = weighted_means.to_pandas().add_prefix("Landtemp_")
    df.index = pd.to_datetime(df.index).to_period("M")
    return df


def open_precipitation():
    dfs = []
    for file in os.scandir("weather/pre/"):
        if file.path.endswith(".nc"):
                dfs.append(xr.open_dataset(file.path))
    ds5 = xr.combine_by_coords(dfs)
    da = ds5.pre
    weighted_means = take_weighted_mean(da)
    df = weighted_means.to_pandas().add_prefix("Pre_")
    df.index = df.index.to_period("M")
    return df


def open_sst_netcdf():
    sea_filename = "weather/temp/HadSST.4.0.1.0_median.nc"
    ds = xr.open_dataset(sea_filename)
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    da = ds.tos
    weighted_means = take_weighted_mean(da)
    df = weighted_means.to_pandas().add_prefix("sst_")
    df.index = pd.to_datetime(df.index).to_period("M")
    return df


def open_sink(model, sink_type):
    type_name = "land" if sink_type=="DGVM" else "ocean"
    # Read the corresponding model data into a df:
    df = pd.read_csv(f'sink/{sink_type}_monthly/{sink_type}_{model}_monthly.csv', 
                       names=["_timestamp",
                              f"{type_name}_sink_global",
                              f"{type_name}_sink_North",
                              f"{type_name}_sink_Tropics",
                              f"{type_name}_sink_South"],
                       delimiter=" ")
    index = pd.to_datetime(df._timestamp, format="%Y%m").rename("time")
    df = df.set_index(index).to_period("M")
    del df['_timestamp']
    return df


def average_land_sink():
    names = ["CLM5.0", "IBIS", "ISAM", "ISBA-CTRIP", "JSBACH",
             "JULES-ES", "LPJ", "LPX-Bern", "OCN", "ORCHIDEEv3",
             "SDGVM", "VISIT", "YIBs"]
    dfs = []
    for name in names:
        df = open_sink(name, "DGVM")
        dfs.append(df)
    mean = pd.concat(dfs).groupby(level=[0]).mean()
    return mean


def average_ocean_sink():
    names = ["cesm", "cnrm", "csiro", "fesom", "ipsl", "mpi", "noresm",
             "planktom", "princeton"]
    dfs = []
    for name in names: 
        df = open_sink(name, "GOBM")
        dfs.append(df)
    mean = pd.concat(dfs).groupby(level=[0]).mean()
    return mean


def open_co2():
    co2 = pd.read_csv('CO2/monthly_mlo_spo.csv')
    co2 = co2.set_index(pd.to_datetime(co2['time'])).to_period("M")
    del co2['time']
    return co2


def open_enso():
    enso = pd.read_csv('weather/ENSO.csv', index_col=0)
    enso= enso.set_index(pd.to_datetime(enso.index)).to_period("M")
    return enso


def open_ffs():
    ffs = pd.read_csv('emissions/GCP-GridFED.csv', index_col=0)
    ffs = ffs.set_index(pd.to_datetime(ffs.index)).to_period("M")
    return ffs


def generate_all_monthly_data():
    df = open_co2().join(
         average_land_sink()).join(
         average_ocean_sink()).join(
         open_landtemp_netcdf()).join(
         open_sst_netcdf()).join(
         open_precipitation()).join(
         open_enso()).join(
         open_ffs())
    df.to_csv('all_data.csv')


def open_all_data():
    return pd.read_csv('core/data/all_data.csv', index_col=[0, 1])


def drop_na(df):
    df_no_na = df.reset_index()
    df_no_na['Date'] = pd.to_datetime(df_no_na[['Year', 'Month']].assign(DAY=1))
    df_no_na = df_no_na.dropna()
    df_no_na = df_no_na.reset_index()
    df_no_na = df_no_na[['Year', 'Month', 'Date'] + [c for c in df_no_na if c not in ['Year', 'Month', 'Date', 'index']]]
    print(f"There are {len(df_no_na)} rows remaining out of an initial {len(df)}.")
    return df_no_na


generate_all_monthly_data()  
