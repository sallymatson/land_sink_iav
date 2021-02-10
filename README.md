## Using ML to estimate carbon sinks
Dissertation project for MSc Climate Change.

## Data Sources
### CO<sub>2</sub> Data
Access to the NOAA ESRL data files is found [here](https://www.esrl.noaa.gov/gmd/ccgg/trends/gl_data.html):
* `co2_mm_gl.csv`: Globally averaged marine surface monthly mean data
* `co2_annmean_gl.csv`: Globally averaged marine surface yearly mean data
* `co2_gr_gl.csv`: Globally averaged marine surface annual mean growth rates

Monthly atmospheric data from MLO and SPO is [here](https://scrippsco2.ucsd.edu/data/atmospheric_co2/mlo.html):
* `monthly_mlo_spo.xlsx`: File sent from Corinne on 2/1/2021.

### Weather Data
Temperature (using [HadCRUT5 data](https://crudata.uea.ac.uk/cru/data/temperature/)):
* `HadCRUT5_monthlymean.txt`: Global monthly mean temperature, downloaded from CRU website
  * `HadCRUT5_mm.csv`: csv version (process using `read_cru_hemi` from `utils.R`)
* `HadCRUT5_nh.txt`: Northern Hemisphere mean temperature
  * `HadCRUT5_nh.csv`: csv version
* `HadCRUT5_sh.txt`: Southern Hemisphere mean temperature
  * `HadCRUT5_sh.csv`: csv version

Precipitation:
* ?

### Carbon Sink Data
* `Global_Carbon_Budget_2020v1.0.xlsx`: [2020 carbon budget](https://www.icos-cp.eu/science-and-impact/global-carbon-budget/2020)
  * `global_sink.csv`: Extracting information from above, column D - column C - column B
* `DGVM_monthly`: Land sink information generated from `cal_DGVM_monthly.py`:
  * `DGVM_MODEL_monthly.csv`: Columns are date (YYYYMM), global, north exatropics, tropics, and south exatropics in PgC.
* `GOBM_monthly`: Ocean sink information generated from `cal_GOBM_monthly.py`:
  * `GOBM_MODEL_monthly.csv`: Columns are date (YYYYMM), global, north exatropics, tropics, and south exatropics in PgC.
