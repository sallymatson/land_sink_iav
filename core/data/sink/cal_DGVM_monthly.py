#= get the variable name automaticaly
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

import netCDF4 as nc4
import numpy as np
from calendar import monthrange
import pandas as pd
import os
from scipy.integrate import quad
from math import sqrt, cos, sin, radians

#=== 1 Reading input files 
#modelname='CABLE' #nodata
#modelname='CLASS-CTEM'#nodata
#modelname='CLM5.0'
#modelname='IBIS' 
modelname='ISAM'
#modelname='ISBA-CTRIP'
#modelname='JSBACH'
#modelname='JULES-ES'
#modelname='LPJ'  
#modelname='LPX-Bern'
#modelname='OCN'
#modelname='ORCHIDEEv3'
#modelname='SDGVM'  
#modelname='VISIT'
#modelname='YIBs'

infile='./DATA/DGVM_'+str(modelname)+'_monthly_11.nc'
f=nc4.Dataset(infile)
fvar=list(f.variables.keys())
if fvar.count('latitude') >0 :
 latname='latitude'
else:
 latname='lat'
if fvar.count('longitude') >0 :
 lonname='longitude'
else:
 lonname='lon'

ym=f.variables['time']
syear=int(ym[0]/100)
if (syear >3000 or syear <1700 ):
 syear=1700  # CABLE/OCN

#=== 2 Extract fluxes
#= land
land=f.variables['nbp']
gland=land[:,:,:]
gland=np.nan_to_num(gland)
gland[(gland==-99999.)]=0 # fill value

#print(np.max(gland),np.min(gland))

ntime=np.shape(gland)[0]

#= lats
lats=f.variables[latname]
latslist=lats[:]
nlat=len(latslist)
lons=f.variables[lonname]
lonslist=lons[:]

#=== 3 Calculate (the global, monthly, monthly average, annual residual)
#=== area
f=nc4.Dataset('./DATA/grid_cell_area11.nc')
tarea=f.variables['grid_cell_area'] #m2
area=tarea[:,:]
fflag=nc4.Dataset('./DATA/Transcom_reg23_11.nc')
tmask=fflag.variables['transcom_regions']
mask=tmask[:,:]
lmask=mask
#== land
lmask[(mask>=1) & (mask<=11)]=1
lmask[(mask>=12)]=0

#= dfm:monthly 
tm=0
rowsm=[]
glandmlist=[]
ldtotalms=[];ldnorthms=[];ldtropicms=[];ldsouthms=[]
for im in np.arange(ntime):
  glandm=gland[im,:,:]
  #print(np.min(glandm),np.max(glandm))
  nyear=im//12 #integer only
  year=syear+nyear
  tm=im+1-12*nyear  # im=12,tm=1
  #print(year,tm)
  day=monthrange(year,tm)
  days=day[1]
  seconds=days*24*3600 #
  area=np.multiply(area,lmask)
  glandmt1=np.multiply(glandm,area)
  #print(np.min(glandmt1),np.max(glandmt1),np.max(area))
  #= all grids of one month
  convert=1e-12*seconds*-1
  ldtotalm=np.sum(glandmt1)*convert
  ldnorthm=np.sum(glandmt1[120:nlat,:])*convert       #>30N
  ldtropicm=np.sum(glandmt1[60:120])*convert    #30-30N
  ldsouthm=np.sum(glandmt1[0:60,:])*convert    #30S

  ldtotalms=np.append(ldtotalms,ldtotalm)
  ldnorthms=np.append(ldnorthms,ldnorthm)
  ldtropicms=np.append(ldtropicms,ldtropicm)
  ldsouthms=np.append(ldsouthms,ldsouthm)

  yearmm=year*100+tm
  print(year,tm,ldtotalm)
  rowm=[yearmm,
               ldtotalm,ldnorthm,ldtropicm,ldsouthm,
               ]
  rowsm.append(rowm)

dfm=pd.DataFrame(rowsm,columns=['yearmm',
                                        'ldtotalm','ldnorthm','ldtropicm','ldsouthm',
                                ])

#=== 4 save data
outfile1='DGVM_'+str(modelname)+'_monthly.csv'
dirpath=os.getcwd()
foldername='/OUT_csv'
dir1=dirpath+foldername
fullpath1=os.path.join(dir1,outfile1)
dfm.to_csv(fullpath1,sep=' ',float_format='%.3f',header=False,index=False)

#=== end of the file


