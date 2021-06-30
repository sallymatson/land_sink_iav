#=== created by zhaohui.chen@uea.ac.uk
#    output : global/north/tropic/south  total/land/ocean  *_monthly.csv
#=   read regridded 1*1 files
#=   all original files were regridded to 1*1 in ../regrid
#=   can be used to read all inversion results
#=== 06Aug2019 (outdated 20191108)
#= 1 Reading satellite inversion results TM3 OCFP *.nc
#= 2 Extract all fluxes
#= 3 Calculate monthly

#=   mean climatology, annual residual)
#= # from Penelope
#= """first subtract the 2009-2015 mean from each monthly value for each grid square
#= This will retain the trend in the data which will then come out in the residuals,
#= which is what Corinne has asked for.
#= Then calculate the mean climatology for the whole 2010-2014 period (excluding spin up and down time
#= Finally, subtract the mean climatology for the whole period from the climatology
#= for each year to get the residual (which will contain trend and IAV information)) """ 

#= S1 monthly mean= 2009-2015 mean
#= S2 monthly = monthly - S1 monthly mean
#= S3 mean climatology for 2010-2014 from S2 results
#= S4 residual= S2 monthly - S3 mean climatology

#= latitude averaged for 2010-2015 

#= 4 Save the results in *.csv 
#= four files dfm(monthly) dfy(annual),dfmcli(avemonthly),dflat(zonally) 
import sys
import os
sys.path.append(os.getcwd())

import netCDF4 as nc4
import numpy as np
from calendar import monthrange
import pandas as pd
import os
from scipy.integrate import quad
from math import sqrt, cos, sin, radians

#=== 1 Reading input files 
model=['cesm','csiro','fesom','mpi','cnrm','ipsl','planktom','noresm','princeton','roeden','land','cmems','csir','watson']
i=12 #0-13
infile='./DATA_2020/GOBCM_'+str(model[i])+'_monthly_11.nc'
f=nc4.Dataset(infile)
ym=f.variables['time']
syear=int(ym[0]/100)

#=== 2 Extract fluxes
#= ocean
flux=f.variables['co2flux_ocean']
gflux=flux[:,:,:]
ntime=np.shape(gflux)[0]

#= lats
lats=f.variables['lat']
latslist=lats[:]
nlat=len(latslist)
lons=f.variables['lon']
lonslist=lons[:]
#=== 3 Calculate monthly
#= replace nan
gflux=np.nan_to_num(gflux)
#= cycle for every month

#=== area
f=nc4.Dataset('./DATA/grid_cell_area11.nc')
tarea=f.variables['grid_cell_area'] #m2
area=tarea[:,:]
fflag=nc4.Dataset('./DATA/Transcom_reg23_11.nc')
tmask=fflag.variables['transcom_regions']
mask=tmask[:,:]
#== ocean
omask=mask
omask[(mask<12)]=0
omask[(mask>=12)& (mask<=22)]=1
omask[(mask>22)]=0


#= dfm:monthly 
tm=0
rowsm=[]
totalms=[];northms=[];tropicms=[];southms=[]
for im in np.arange(ntime):
  gfluxm=gflux[im,:,:]
  nyear=im//12 #integer only
  year=syear+nyear
  tm=im+1-12*nyear  # im=12,tm=1
  print(year,tm)
  day=monthrange(year,tm)
  days=day[1]
  seconds=days*24*3600 #
  area=np.multiply(area,omask)
  gfluxmt1=np.multiply(gfluxm,area)

  #= all grids of one month
  totalm=np.sum(gfluxmt1)*1e-12*seconds*-1
  northm=np.sum(gfluxmt1[120:nlat,:])*1e-12*seconds*-1       #>30N
  tropicm=np.sum(gfluxmt1[60:120])*1e-12*seconds*-1    #30-30N
  southm=np.sum(gfluxmt1[0:60,:])*1e-12*seconds*-1    #30S

  totalms=np.append(totalms,totalm)
  northms=np.append(northms,northm)
  tropicms=np.append(tropicms,tropicm)
  southms=np.append(southms,southm)

  yearmm=year*100+tm
  rowm=[yearmm,
               totalm,northm,tropicm,southm,
               ]
  rowsm.append(rowm)

dfm=pd.DataFrame(rowsm,columns=['yearmm',
                                        'totalm','northm','tropicm','southm',
                                ])

#=== 4 save data
outfile1='GOBCM_'+str(model[i])+'_monthly.csv'
dirpath=os.getcwd()
foldername='/OUT_csv'
dir1=dirpath+foldername
fullpath1=os.path.join(dir1,outfile1)
dfm.to_csv(fullpath1,sep=' ',float_format='%.3f',header=False,index=False)

#=== end of the file


