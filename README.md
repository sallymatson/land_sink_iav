## Using ML to estimate carbon sinks
Dissertation project for MSc Climate Change. 

- 3/17/2021: Dissertation proposal submitted
- 3/31/2021: [Progress presentation](https://docs.google.com/presentation/d/1cRt70FpyEDJgw5rioyKlGI4nsppwOjQM4TiY3B94d3Q/edit) to lab group

## Data Sources
	* [Monthly atmospheric data from MLO and SPO](https://scrippsco2.ucsd.edu/data/atmospheric_co2/mlo.html):
		* `monthly_mlo_spo.xlsx`: File sent from Corinne on 2/1/2021.
		* `monthly_mlo_spo.csv`: Extraction from the excel file.
* Emissions: `emissions/GCP-GRidFED.csv` (From Matt)


| Dataset                | Timescale      | Units      |
| ---------------------- | -------------- | ---------- |
| [CO2 (NOAA ESRL)](https://www.esrl.noaa.gov/gmd/ccgg/trends/gl_data.html)| 1980 - 11/2020 | ppm        |
| [CO2 (MLO, SPO)](https://scrippsco2.ucsd.edu/data/atmospheric_co2/mlo.html)         | 1958 - 10/2020 | GtC/month  |
| [Land temp (CRUTEM5)](https://crudata.uea.ac.uk/cru/data/temperature/) | 1850 - 2019    | anomaly in degC    |
| [SST (HadSST4)](https://crudata.uea.ac.uk/cru/data/temperature/) | 1850 - 2019    | anomaly in degC    |
| [Precipitation (TS4)](https://catalogue.ceda.ac.uk/uuid/89e1e34ec3554dc98594a5732622bce9)    | 1901 - 12/2019 | mm/month   |
| [Wind speed](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.derived.surface.html)    | ? | m/s  |
| Global carbon budget   | 1959 - 2019    |            |
| DGVMs                  | 1700 - 12/2019 | GtC/month  |
| GCBMs                  | 1958 - 12/2019 | GtC/month  |
| FF emissions           | 1959 - 11/2019 | kgCO2/year |

