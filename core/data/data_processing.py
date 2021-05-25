import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def iav_resid_only(df, variables=None):
    '''
    Returns the residual for each variable after removing seasonal and overall trend. 
    Uses an additive naive trend; first removes overall using moving window, then seasonal. 
    More info: https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
    '''
    df_iav = df.copy(deep=True)
    if variables is None:
        variables = [var for var in df.columns if var not in ['Month', 'Year', 'Date', 'ENSO']]
    for var in variables:
        decompose_result = seasonal_decompose(df[var], model="additive", period=12, extrapolate_trend=1)
        df_iav[var] = decompose_result.resid
    return df_iav


def iav_with_trend(df, variables=None):
    df_iav = df.copy(deep=True)
    if variables is None:
        variables = [var for var in df.columns if var not in ['Month', 'Year', 'Date', 'ENSO']]
    for var in variables:
        decompose_result = seasonal_decompose(df[var], model="additive", period=12, extrapolate_trend=1)
        df_iav[var] = decompose_result.resid + decompose_result.trend
    return df_iav


def trend(df, variables=None):
    df_iav = df.copy(deep=True)
    if variables is None:
        variables = [var for var in df.columns if var not in ['Month', 'Year', 'Date', 'ENSO']]
    for var in variables:
        decompose_result = seasonal_decompose(df[var], model="additive", period=12, extrapolate_trend=1)
        df_iav[var] = decompose_result.trend
    return df_iav