import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def open_precipitation():

    ds1 = xr.open_dataset("data/weather/cru_ts4.04.1981.1990.pre.dat.nc")
    ds2 = xr.open_dataset("data/weather/cru_ts4.04.1991.2000.pre.dat.nc")
    ds3 = xr.open_dataset("data/weather/cru_ts4.04.2001.2010.pre.dat.nc")
    ds4 = xr.open_dataset("data/weather/cru_ts4.04.2011.2019.pre.dat.nc")
    ds5 = xr.combine_by_coords([ds1, ds2, ds3, ds4])
    # Global monthly means:
    means = ds5.pre.mean(['lon', 'lat'])
    # Monthly means by region:
    bins = ds5.pre.groupby_bins("lat", [-90, -30, 30, 90], labels=["pre_South", "pre_Tropics", "pre_North"]).mean(['lat', 'lon'])
    # Create df:
    tuples = list(zip(means.to_dataframe().index.year, means.to_dataframe().index.month))
    index = pd.MultiIndex.from_tuples(tuples, names=["Year", "Month"])
    df = pd.DataFrame('-', index, ['Global_mean', 'Tropics'])
    df['pre_Global_mean'] = means
    df[['pre_South', 'pre_Tropics', 'pre_North']] = bins.values
    '''
    #Precipitation over time:
    bins.plot.line(x='time')
    plt.show()

    # 2d plots:
    ds2d = ds.pre.isel(time=50)
    ds2d.plot()
    plt.show() 
    '''
    return df


def open_cru_file(path):
    '''
    Reads a file in CRU format (https://crudata.uea.ac.uk/cru/data/temperature/#filfor)
    into a multiindex pandas dataframe (indecies are year & month.)
    '''
    years = range(1850, 2021)
    months = range(1, 13)
    index = pd.MultiIndex.from_product([years, months], names=['Year', 'Month'])
    df = pd.DataFrame('-', index, ['Temp', 'Coverage'])
    with open(path, "r") as f:
        for year in years:
            temp_line = f.readline().strip().split()
            coverage_line = f.readline().split()
            df.loc[year]['Temp'] = temp_line[1:-1]
            df.loc[year]['Coverage'] = coverage_line[1:]   
    return df


def open_temperature():

    temp_gl = open_cru_file('data/weather/HadCRUT5_gl.txt')
    temp_nh = open_cru_file('data/weather/HadCRUT5_nh.txt')
    temp_sh = open_cru_file('data/weather/HadCRUT5_sh.txt')
    sh_nh = temp_sh.merge(temp_nh, on=['Year', 'Month'], suffixes=["_sh", "_nh"])
    df = temp_gl.merge(sh_nh, on=['Year', 'Month'])

    return df


def open_co2():

    monthlymean = pd.read_csv('data/CO2/co2_mm_gl.csv', skiprows=56, index_col=[0, 1])
    mlo_spo = pd.read_csv('data/CO2/monthly_mlo_spo.csv', index_col=[0, 1])
    overall = mlo_spo.join(monthlymean)

    return overall


def open_sink(model="CLM5.0"):

    dgvm = pd.read_csv(f'data/sink/DGVM_monthly/DGVM_{model}_monthly.csv', 
                       names=["_timestamp", "sink_global", "sink_North", "sink_Tropics", "sink_South"],
                       delimiter=" ")
    dgvm['Year'] = [int(str(d)[0:4]) for d in dgvm['_timestamp']]
    dgvm['Month'] = [int(str(d)[4:6].lstrip('0')) for d in dgvm['_timestamp']]
    dgvm = dgvm.set_index(['Year', 'Month'])
    del dgvm['_timestamp']

    return dgvm


def yearly_data():

    co2_ppm = pd.read_csv('data/CO2/co2_annmean_gl.csv', skiprows=55, index_col=0)
    co2_gr = pd.read_csv('data/CO2/co2_gr_gl.csv', skiprows=60, index_col=0)
    sink_gr = pd.read_csv('data/sink/annual_global_sink.csv', index_col=0)
    df = sink_gr.merge(co2_gr.merge(co2_ppm, on='Year', how='left'), on='Year')


def get_all_monthly_data():

    co2 = open_co2()
    sink = open_sink()
    pre = open_precipitation()
    temp = open_temperature()
    df = co2.join(pre.join(temp.join(sink)))
