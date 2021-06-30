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

    # Weighted global mean
    weighted_gl = da.weighted(weights)
    gl_mean = weighted_gl.mean(('lon', 'lat'))
    gl_mean.name = "GL"

    # For the three regions, take the weighted averages
    s_ext = da.sel(lat=slice(-90, -30))
    weighted_s_ext = s_ext.weighted(weights)
    s_ext_mean = weighted_s_ext.mean(['lon', 'lat'], skipna=True)
    s_ext_mean.name = "SHex"

    # tropics
    tropics = da.sel(lat=slice(-30, 30))
    weighted_tropics = tropics.weighted(weights)
    tropics_mean = weighted_tropics.mean(['lon', 'lat'], skipna=True)
    tropics_mean.name = "TR"

    # n ext
    n_ext = da.sel(lat=slice(30, 90))
    weighted_n_ext = n_ext.weighted(weights)
    n_ext_mean = weighted_n_ext.mean(['lon', 'lat'], skipna=True)
    n_ext_mean.name = "NHex"
    # Combine into one ds
    return xr.merge([gl_mean, s_ext_mean, tropics_mean, n_ext_mean])


def open_wind(package_dir):
    wind_filename = os.path.join(package_dir,"weather/wind/wspd.sig995.mon.mean.nc")
    ds = xr.open_dataset(wind_filename)
    da = ds.wspd
    da = da.reindex(lat=list(reversed(da.lat)))  # lat s->n to match others
    weighted_means = take_weighted_mean(da)
    df = weighted_means.to_dataframe().add_prefix("WSPD_")
    df.index = df.index.to_period("M")
    return df


def open_landtemp_netcdf(package_dir):
    land_filename = os.path.join(package_dir,"weather/temp/CRUTEM.5.0.1.0.alt.anomalies.nc")
    ds = xr.open_dataset(land_filename)
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    da = ds.tas
    weighted_means = take_weighted_mean(da)
    df = weighted_means.to_dataframe().add_prefix("Landtemp_")
    df.index = pd.to_datetime(df.index).to_period("M")
    return df


def open_precipitation(package_dir):
    dfs = []
    for file in os.scandir(os.path.join(package_dir,"weather/pre/")):
        if file.path.endswith(".nc"):
                dfs.append(xr.open_dataset(file.path))
    ds5 = xr.combine_by_coords(dfs)
    da = ds5.pre
    weighted_means = take_weighted_mean(da)
    df = weighted_means.to_dataframe().add_prefix("Pre_")
    df.index = df.index.to_period("M")
    return df


def open_sst_netcdf(package_dir):
    sea_filename = os.path.join(package_dir,"weather/temp/HadSST.4.0.1.0_median.nc")
    ds = xr.open_dataset(sea_filename)
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    da = ds.tos
    weighted_means = take_weighted_mean(da)
    df = weighted_means.to_dataframe().add_prefix("sst_")
    df.index = pd.to_datetime(df.index).to_period("M")
    return df


def open_sink(model, sink_type, package_dir):
    type_name = "land" if sink_type=="DGVM" else "ocean"
    # Read the corresponding model data into a df:
    df = pd.read_csv(os.path.join(package_dir,f'sink/{sink_type}_monthly/{sink_type}_{model}_monthly.csv'), 
                       names=["_timestamp",
                              f"{type_name}_sink_GL",
                              f"{type_name}_sink_NHex",
                              f"{type_name}_sink_TR",
                              f"{type_name}_sink_SHex"],
                       delimiter=" ")
    index = pd.to_datetime(df._timestamp, format="%Y%m").rename("time")
    df = df.set_index(index).to_period("M")
    del df['_timestamp']
    return df


def average_land_sink(package_dir):
    names = ["CLM5.0", "IBIS", "ISAM", "ISBA-CTRIP", "JSBACH",
             "JULES-ES", "LPJ", "LPX-Bern", "OCN", "ORCHIDEEv3",
             "SDGVM", "VISIT", "YIBs"]
    dfs = []
    for name in names:
        df = open_sink(name, "DGVM", package_dir)
        dfs.append(df)
    mean = pd.concat(dfs).groupby(level=[0]).mean()
    return mean


def average_ocean_sink(package_dir):
    names = ["cesm", "cnrm", "csiro", "fesom", "ipsl", "mpi", "noresm",
             "planktom", "princeton"]
    dfs = []
    for name in names: 
        df = open_sink(name, "GOBM", package_dir)
        dfs.append(df)
    mean = pd.concat(dfs).groupby(level=[0]).mean()
    std = pd.concat(dfs).groupby(level=[0]).std()
    return mean, std


def all_dgvms_gl(package_dir):
    names = ["CLM5.0", "IBIS", "ISAM", "ISBA-CTRIP", "JSBACH",
             "JULES-ES", "LPJ", "LPX-Bern", "OCN", "ORCHIDEEv3",
             "SDGVM", "VISIT", "YIBs"]
    dfs = []
    for name in names:
        df = open_sink(name, "DGVM", package_dir)
        df[name] = df['land_sink_GL']
        dfs.append(df[[name]])
    return pd.concat(dfs, axis=1)


def open_co2(package_dir):
    co2 = pd.read_csv(os.path.join(package_dir,'CO2/monthly_mlo_spo.csv'))
    co2 = co2.set_index(pd.to_datetime(co2['time'])).to_period("M")
    del co2['time']
    return co2


def open_enso(package_dir):
    enso = pd.read_csv(os.path.join(package_dir,'weather/ENSO.csv'), index_col=0)
    enso= enso.set_index(pd.to_datetime(enso.index)).to_period("M")
    return enso


def open_ffs(package_dir):
    ffs = pd.read_csv(os.path.join(package_dir,'emissions/GCP-GridFED.csv'), index_col=0)
    ffs = ffs.set_index(pd.to_datetime(ffs.index)).to_period("M")
    return ffs


'''
OPENING DATA

'''


def generate_all_monthly_data():
    package_dir = os.path.dirname(os.path.abspath(__file__))
    average_ocean_sink_val, _ = average_ocean_sink(package_dir)
    df = open_co2(package_dir).join(
         average_land_sink(package_dir)).join(
         average_ocean_sink_val).join(
         open_landtemp_netcdf(package_dir)).join(
         open_sst_netcdf(package_dir)).join(
         open_precipitation(package_dir)).join(
         open_wind(package_dir)).join(
         open_enso(package_dir)).join(
         open_ffs(package_dir))
    df.to_csv('all_data.csv')


def open_all_data():
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(package_dir,'all_data.csv'), index_col=0)


'''
INFERRED LAND SINK
'''

def open_global_ffs(package_dir):
    ffs = pd.read_csv(os.path.join(package_dir,'emissions/GCP-GridFED-GL.csv'), index_col=0, skiprows=2)
    ffs = ffs.set_index(pd.to_datetime(ffs.index)).to_period("M")
    return ffs


def open_luc(package_dir):
    luc = pd.read_csv(os.path.join(package_dir,'LUC/global_LUC.csv'), index_col=0)
    luc = luc.set_index(pd.to_datetime(luc.index, format="%Y")).to_period("M")
    luc = luc.resample("M").pad()/12
    return luc


def open_co2_monthly_gr(package_dir):
    co2_gr = pd.read_csv(os.path.join(package_dir,'CO2/co2_mm_gl.csv'), skiprows=59, index_col=0)
    co2_gr = co2_gr.set_index(pd.to_datetime(co2_gr.index, format="%Y-%m")).to_period("M")
    return co2_gr[['monthly_gr']]


def inferred_land_sink():
    # Inferred Land sink = FF + LUC - AGR - ocean sink
    package_dir = os.path.dirname(os.path.abspath(__file__))
    ff = open_global_ffs(package_dir) # GtC / month
    luc = open_luc(package_dir) # GtC / month
    mgr = open_co2_monthly_gr(package_dir) # GtC / month
    emission_flux = ff + luc
    ocean_sink, _ = average_ocean_sink(package_dir) # GtC / month
    ocean_sink = ocean_sink[['ocean_sink_GL']]
    all_data = ff.join(luc).join(mgr).join(ocean_sink)
    return all_data


def inferred_land_sink_uncertainty():
    # Combine errors ~ in quadrature ~
    # http://ipl.physics.harvard.edu/wp-uploads/2013/03/PS3_Error_Propagation_sp13.pdf
    package_dir = os.path.dirname(os.path.abspath(__file__))
    #package_dir = 'core/data/'
    # CO2: use annual GR uncertainty, distribute evenly over year
    co2 = pd.read_csv(os.path.join(package_dir,'CO2/co2_gr_gl.csv'), skiprows=60, index_col=0)
    co2 = co2.set_index(pd.to_datetime(co2.index, format="%Y")).to_period("M")
    co2 = co2.resample("M").pad()
    co2_unc = co2[['co2_gr_unc_monthly']]
    co2_unc = co2_unc*2.12  # convert from ppm to gt/c
    # FF: 5% of annual, so 1.44% of annual each month (calculated in quadrature).
    # Thus, assuming the error is distributed proportionately to monthly val, each month
    # is 1.44% * 12 (to scale up, bc 1.44% is calucated monthly) i.e. 17%
    ff = open_global_ffs(package_dir)
    ff_unc = ff*0.17
    # Ocean sink: std of models
    ocean, ocean_unc = average_ocean_sink(package_dir)
    ocean_unc = ocean_unc[['ocean_sink_GL']]
    unc = ff.join(co2_unc).join(ocean_unc)
    # From GCB, yearly LUC unc is +/- 0.7 GtC/year
    # Monthly is 0.202 using quadrature
    unc['LUC'] = 0.202
    unc = unc.dropna()
    # Square all values; sum across each row; take square root
    unc = (unc**2).sum(axis=1) ** 1/2
    unc.to_csv('Inferred_land_sink_unc.csv')
