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
    temp_gl = open_cru_file('weather/temp/HadCRUT5_gl.txt')
    temp_gl.columns = ['Temp_gl', 'Coverage_gl']
    temp_nh = open_cru_file('weather/temp/HadCRUT5_nh.txt')
    temp_sh = open_cru_file('weather/temp/HadCRUT5_sh.txt')
    sh_nh = temp_sh.merge(temp_nh, on=['Year', 'Month'], suffixes=["_sh", "_nh"])
    df = temp_gl.merge(sh_nh, on=['Year', 'Month'])
    del df['Coverage_gl']
    del df['Coverage_sh']
    del df['Coverage_nh']
    return df


def yearly_data():
    co2_ppm = pd.read_csv('CO2/co2_annmean_gl.csv', skiprows=55, index_col=0)
    co2_gr = pd.read_csv('CO2/co2_gr_gl.csv', skiprows=60, index_col=0)
    sink_gr = pd.read_csv('sink/annual_global_sink.csv', index_col=0)
    df = sink_gr.merge(co2_gr.merge(co2_ppm, on='Year', how='left'), on='Year')
    return(df)
