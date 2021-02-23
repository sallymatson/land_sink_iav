import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def open_precipitation():
    '''
    Returns a df with precipitation for the tropics, south & north exatropics spanning 1981-2019
    '''
    ds1 = xr.open_dataset("data/weather/cru_ts4.04.1981.1990.pre.dat.nc")
    ds2 = xr.open_dataset("data/weather/cru_ts4.04.1991.2000.pre.dat.nc")
    ds3 = xr.open_dataset("data/weather/cru_ts4.04.2001.2010.pre.dat.nc")
    ds4 = xr.open_dataset("data/weather/cru_ts4.04.2011.2019.pre.dat.nc")
    ds5 = xr.combine_by_coords([ds1, ds2, ds3, ds4])
    # Global monthly means:
    means = ds5.pre.mean(['lon', 'lat'])
    # Monthly means by latitude region:
    bins = ds5.pre.groupby_bins("lat", [-90, -30, 30, 90], labels=["Pre_south", "Pre_tropics", "Pre_north"]).mean(['lat', 'lon'])
    # Create & populate df:
    tuples = list(zip(means.to_dataframe().index.year, means.to_dataframe().index.month))
    index = pd.MultiIndex.from_tuples(tuples, names=["Year", "Month"])
    df = pd.DataFrame('-', index, [])
    df['Pre_global'] = means
    df[['Pre_South', 'Pre_Tropics', 'Pre_North']] = bins.values
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
    '''
    Returns a df with the average monthly temperature globally & for NH/SH.
    '''
    temp_gl = open_cru_file('data/weather/HadCRUT5_gl.txt')
    temp_gl.columns = ['Temp_gl', 'Coverage_gl']
    temp_nh = open_cru_file('data/weather/HadCRUT5_nh.txt')
    temp_sh = open_cru_file('data/weather/HadCRUT5_sh.txt')
    sh_nh = temp_sh.merge(temp_nh, on=['Year', 'Month'], suffixes=["_sh", "_nh"])
    df = temp_gl.merge(sh_nh, on=['Year', 'Month'])
    return df


def open_co2():
    monthlymean = pd.read_csv('data/CO2/co2_mm_gl.csv', skiprows=56, index_col=[0, 1])
    del monthlymean['decimal']
    del monthlymean['trend']
    monthlymean.columns = ['CO2_ppm_gl']
    mlo_spo = pd.read_csv('data/CO2/monthly_mlo_spo.csv', index_col=[0, 1])
    overall = mlo_spo.join(monthlymean)
    return overall


def open_sink(model="CLM5.0", sink_type="DGVM"):
    df = pd.read_csv(f'data/sink/{sink_type}_monthly/{sink_type}_{model}_monthly.csv', 
                       names=["_timestamp", "sink_global", "sink_North", "sink_Tropics", "sink_South"],
                       delimiter=" ")
    df['Year'] = [int(str(d)[0:4]) for d in df['_timestamp']]
    df['Month'] = [int(str(d)[4:6].lstrip('0')) for d in df['_timestamp']]
    df = df.set_index(['Year', 'Month'])
    del df['_timestamp']
    return df


def average_dgvms():
    dgvms = ["CLM5.0", "IBIS", "ISAM", "ISBA-CTRIP", "JSBACH", "JULES", "LPJ", "LPX", "OCN", "ORCHIDEEv3", "SDGVM", "VISIT", "YIBs"]
    for dgvm in dgvms:
        df = open_sink(dgvm, "DGVM")


def yearly_data():
    co2_ppm = pd.read_csv('data/CO2/co2_annmean_gl.csv', skiprows=55, index_col=0)
    co2_gr = pd.read_csv('data/CO2/co2_gr_gl.csv', skiprows=60, index_col=0)
    sink_gr = pd.read_csv('data/sink/annual_global_sink.csv', index_col=0)
    df = sink_gr.merge(co2_gr.merge(co2_ppm, on='Year', how='left'), on='Year')
    return(df)


def generate_all_monthly_data():
    co2 = open_co2()
    sink = open_sink()
    pre = open_precipitation()
    temp = open_temperature()
    df = co2.join(pre.join(temp.join(sink)))
    df.to_csv('data/all_data.csv')


def open_all_data():

    return pd.read_csv('data/all_data.csv', index_col=[0, 1])
