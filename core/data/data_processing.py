import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def trend(df, variables=None):
    df_iav = df.copy(deep=True)
    if variables is None:
        variables = [var for var in df.columns if var not in ['Month', 'Year', 'Date', 'ENSO']]
    for var in variables:
        decompose_result = seasonal_decompose(df[var], model="additive", period=12, extrapolate_trend=1, two_sided=False)
        df_iav[var] = decompose_result.trend
    return df_iav