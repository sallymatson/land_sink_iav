## Using ML to estimate carbon sinks
Dissertation project for MSc Climate Change.

## Data Sources
### CO<sub>2</sub> Data
[NOAA ESRL data](https://www.esrl.noaa.gov/gmd/ccgg/trends/gl_data.html):
* `co2_mm_gl.csv`: Globally averaged marine surface monthly mean data
* `co2_annmean_gl.csv`: Globally averaged marine surface yearly mean data
* `co2_gr_gl.csv`: Globally averaged marine surface annual mean growth rates

[Monthly atmospheric data from MLO and SPO](https://scrippsco2.ucsd.edu/data/atmospheric_co2/mlo.html):
* `monthly_mlo_spo.xlsx`: File sent from Corinne on 2/1/2021.
* `monthly_mlo_spo.csv`: Extraction from the excel file.

### Weather Data
Temperature (using [HadCRUT5](https://crudata.uea.ac.uk/cru/data/temperature/)):
* `HadCRUT5_gl.txt`, `HadCRUT5_nh.txt`, `HadCRUT5_nh.txt`: Global (or North/Southern hemisphere) monthly mean temperature downloaded from CRU website (to open use `data_utils.open_cru_file()`.)

Precipitation (using [CRU TS4 data](https://catalogue.ceda.ac.uk/uuid/89e1e34ec3554dc98594a5732622bce9)):
* `cru_ts4.04.YYYY.YYYY.pre.dat.nc`: time span is first YYYY to second YYYY. Combined using `data_utils.open_precipitation()`.

### Carbon Sink Data
* `Global_Carbon_Budget_2020v1.0.xlsx`: [2020 carbon budget](https://www.icos-cp.eu/science-and-impact/global-carbon-budget/2020)
  * `annual_global_sink.csv`: Extracting relevant information from above.
* `DGVM_monthly`: Land sink information generated from `cal_DGVM_monthly.py`:
  * `DGVM_MODEL_monthly.csv`: Columns are date (YYYYMM), global, north exatropics, tropics, and south exatropics in PgC.
* `GOBM_monthly`: Ocean sink information generated from `cal_GOBM_monthly.py`:
  * `GOBM_MODEL_monthly.csv`: Columns are date (YYYYMM), global, north exatropics, tropics, and south exatropics in PgC.


| Dataset                | Timescale      | Units      |
| ---------------------- | -------------- | ---------- |
| CO2 (NOAA ESRL)        | 1980 - 11/2020 | ppm        |
| CO2 (MLO, SPO)         | 1958 - 10/2020 | GtC/month  |
| Temperature (HADCRUT5) | 1850 - 2019    | anomaly    |
| Precipitation (TS4)    | 1901 - 12/2019 | mm/month   |
| Global carbon budget   | 1959 - 2019    |            |
| DGVMs                  | 1700 - 12/2019 | GtC/month  |
| GCBMs                  | 1958 - 12/2019 | GtC/month  |


### TODOs
* Check how the `.mean()` function deals with NA values for precpitation averages. It should ideally ignore NAs and not count them in the denominator.
* Create a function to average all of the DGVM and GOBMs and add these to the overall sheet.
