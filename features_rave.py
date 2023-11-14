import pandas as pd
pd.set_option('display.max_rows', None)
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import path
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import numpy as np
import netCDF4 as nc
np.set_printoptions(threshold=100000)
from shapely.geometry import Polygon, Point, MultiPoint, LineString, LinearRing
from shapely.ops import cascaded_union, unary_union, transform
import datetime
import math
from scipy.ndimage.interpolation import shift
import scipy.interpolate as si
import shapely.wkt
from shapely.validation import explain_validity
import xarray as xr
import seaborn as sns
from my_functions import sat_vap_press, vap_press, hot_dry_windy, haines
from joblib import Parallel, delayed
import multiprocessing
from os.path import exists
import rasterio
from rasterio.windows import get_data_window,Window, from_bounds
from rasterio.plot import show
from itertools import product

from timezonefinder import TimezoneFinder
import pytz

from helper_functions import build_one_gridcell, calculate_intersection, calculate_grid_cell_corners, make_file_namelist, generate_df

def rave_timeseries(df, day_start_hour, sum_interval, use_weights):
    varis = ['day','FRP_MEAN']#, 'FRP_SD', 'FRE']#, 'CO2', 'CO', 'SO2', 'OC','BC', 'PM25', 'NOx', 'NH3','TPM', 'VOCs', 'CH4'] #don't need 'area', it's the area of each cell
    df_rave = generate_df(varis, len(df))

    #do the intersection, in parallel
    rave_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],'RAVE_GRID_3KM',3000) 
                                 for ii in range(len(df)))
    print([rave_intersections[jj]['weights'].sum() for jj in range(len(rave_intersections))])

    fire_rave_intersection=gpd.GeoDataFrame(pd.concat(rave_intersections, ignore_index=True))
    fire_rave_intersection.set_geometry(col='geometry')    
    #print(fire_rave_intersection)
    fire_rave_intersection = fire_rave_intersection.set_index([str(day_start_hour)+ 'Z Start Day', 'row', 'col'])
    fire_rave_intersection=fire_rave_intersection[~fire_rave_intersection.index.duplicated()]

    fire_rave_intersection_xr = fire_rave_intersection.to_xarray()
    fire_rave_intersection_xr['weights_mask'] = xr.where(fire_rave_intersection_xr['weights']>0,1, np.nan)

    #load in rave data associated with the fire
    times = pd.date_range(np.datetime64(df[str(day_start_hour)+ 'Z Start Day'].iloc[0]),
                        np.datetime64(df[str(day_start_hour)+ 'Z Start Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))
    rave_filenames,times_back_used = make_file_namelist(times,
                                                        '/data2/lthapa/YYYY/RAVE/MM/RAVE-HrlyEmiss-3km-CONUS_v1r1_blend_sYYYYMMDD.nc')                                                 
    
    #print(rave_filenames)
    dat_rave = xr.open_mfdataset(rave_filenames,concat_dim='time',combine='nested',compat='override', coords='all')

    dat_rave = dat_rave.resample(time=str(sum_interval)+'H',base=day_start_hour).sum(dim='time') #take the daily sum
    
    #select the locations and times we want
    dat_rave_sub = dat_rave.isel(grid_yt = fire_rave_intersection_xr['row'].values.astype(int), 
                    grid_xt = fire_rave_intersection_xr['col'].values.astype(int)).sel(
                    time = pd.to_datetime(fire_rave_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values+
                                         'T12:00:00.000000000'))#these should be lined up correctly
    ndays = len(fire_rave_intersection_xr[str(day_start_hour)+ 'Z Start Day'])

    df_rave['day'].iloc[:] = pd.to_datetime(fire_rave_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)
    for var in varis[1:]:
        if use_weights==True:
            df_rave[var] = np.nansum(fire_rave_intersection_xr['weights'].values*dat_rave_sub[var].values, axis=(1,2))
        else:
            df_rave[var] = np.nansum(fire_rave_intersection_xr['weights_mask'].values*dat_rave_sub[var].values,axis=(1,2))
    
    return df_rave


path_poly = '/data2/lthapa/ML_daily/fire_polygons/'
suffix_poly = 'Z_Day_Start.geojson'
start_time=12

#years =[2019,2020]
years = [2021]



for jj in range(len(years)):
    print(path_poly+'ClippedFires'+str(years[jj])+'_VIIRS_daily_'+str(start_time)+suffix_poly)
    
    fire_daily = gpd.read_file(path_poly+'ClippedFires'+str(years[jj])+'_VIIRS_daily_'+str(start_time)+suffix_poly)
    print(fire_daily.crs)
    
    fire_daily=fire_daily.drop(columns=['Current Overpass'])
    fire_daily = fire_daily.drop(np.where(fire_daily['geometry']==None)[0])
    fire_daily['fire area (ha)'] = fire_daily['geometry'].area/10000 #hectares. from m2
    fire_daily.set_geometry(col='geometry', inplace=True) #designate the geometry column
    fire_daily = fire_daily.rename(columns={'Current Day':'UTC Day', 'Local Day': str(start_time)+ 'Z Start Day'})
    
    irwinIDs = np.unique(fire_daily['irwinID'].values)
    print('We are processing ' +str(len(irwinIDs)) + ' unique fires for '+ str(years[jj]))
    
    #smops_all = pd.DataFrame()
    rave_all = pd.DataFrame()
    for ii in range(len(irwinIDs)):
        print(ii)
        fireID=irwinIDs[ii]
        print(fireID)
        df_fire = fire_daily[fire_daily['irwinID']==fireID] #this is what gets fed to the feature selection code
        
        #maybe we filter 12Z Start dates to where we have data?
        days=np.array(df_fire['12Z Start Day'].values, dtype='datetime64')
        df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                              (days<=np.datetime64(str(years[jj]+1)+'-01-01'))].reset_index(drop=True)
        
        #RAVE (always unweighted SUM)
        rave = rave_timeseries(df_fire, start_time, 24, False)
        rave = pd.concat([rave, pd.DataFrame({'irwinID':[fireID]*len(rave)})], axis=1)
        print(rave)
        rave_all = pd.concat([rave_all, rave], axis=0).reset_index(drop=True)
        
    print(rave_all)
    rave_all.to_csv('./fire_features_3/'+'ClippedFires'+str(years[jj])+'_Daily_RAVE_Unweighted_'+str(start_time)+'Z_day_start.csv') #daily sums

