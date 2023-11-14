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

def esi_timeseries(df, day_start_hour):
    #preallocate space for the output
    df_esi_weighted = pd.DataFrame({'day':np.zeros(len(df)),'ESI':np.zeros(len(df))})
    df_esi_unweighted = pd.DataFrame({'day':np.zeros(len(df)),'ESI':np.zeros(len(df))})

    
    #do the intersection, in parallel
    esi_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df_fire.iloc[ii:ii+1],'ESI_GRID',5000) 
                                 for ii in range(len(df_fire)))
    print([esi_intersections[jj]['weights'].sum() for jj in range(len(esi_intersections))])

    
    fire_esi_intersection=gpd.GeoDataFrame(pd.concat(esi_intersections, ignore_index=True))
    fire_esi_intersection.set_geometry(col='geometry')
    
    
    fire_esi_intersection = fire_esi_intersection.set_index([str(day_start_hour)+ 'Z Start Day', 'lat', 'lon'])
    
    fire_esi_intersection_xr = fire_esi_intersection.to_xarray()
    fire_esi_intersection_xr['weights_mask'] = xr.where(fire_esi_intersection_xr['weights']>0,1, np.nan)

    #load in esi data associated with the fire
    times = pd.date_range(np.datetime64(df[str(day_start_hour)+ 'Z Start Day'].iloc[0])-np.timedelta64(1,'W'),
                        np.datetime64(df[str(day_start_hour)+ 'Z Start Day'].iloc[len(df)-1])+np.timedelta64(1, 'W')+
                        np.timedelta64(1,'D'))
    esi_filenames, esi_times = make_file_namelist(times,'/data2/lthapa/YYYY/ESI/DFPPM_4WK_YYYYJJJ.nc')
    
    print(esi_filenames)
    print(esi_times)
    
    
    #open the esi files
    dat_esi = xr.open_mfdataset(esi_filenames,concat_dim='time',combine='nested',compat='override', coords='all')
    dat_esi = dat_esi.assign_coords({'time': esi_times}) #assign coords so we can resample along time
    dat_esi = dat_esi.where(dat_esi['Band1']!=-9999) #gets rid of the missing values!
    dat_esi_daily = dat_esi.reindex(time=times,method='nearest') #makes the weekly data daily
    dat_esi_daily_sub = dat_esi_daily.sel(lat = fire_esi_intersection_xr['lat'].values, 
                                          lon = fire_esi_intersection_xr['lon'].values,
                      time = pd.to_datetime(fire_esi_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values), method='nearest')
                                          


    df_esi_weighted['day'].iloc[:] = pd.to_datetime(fire_esi_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)
    df_esi_unweighted['day'].iloc[:] = pd.to_datetime(fire_esi_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)

    varis=['ESI']
    for var in varis:
        df_esi_weighted[var] = np.nansum(fire_esi_intersection_xr['weights'].values*dat_esi_daily_sub['Band1'].values, axis=(1,2))
        df_esi_unweighted[var] = np.nanmean(fire_esi_intersection_xr['weights_mask'].values*dat_esi_daily_sub['Band1'].values, axis=(1,2))
    
    return df_esi_weighted, df_esi_unweighted
    
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
    

    esi_weighted_all = pd.DataFrame() 
    esi_unweighted_all = pd.DataFrame()
    
    for ii in range(len(irwinIDs)):
        print(ii)
        fireID=irwinIDs[ii]
        print(fireID)
        df_fire = fire_daily[fire_daily['irwinID']==fireID] #this is what gets fed to the feature selection code
        
        #maybe we filter 12Z Start dates to where we have data?
        days=np.array(df_fire['12Z Start Day'].values, dtype='datetime64')
        #df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
        #                      (days<=np.datetime64(str(years[jj]+1)+'-01-01'))].reset_index(drop=True)
        df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                              (days<=np.datetime64(str(years[jj])+'-12-24'))].reset_index(drop=True)
        #PWS
        if len(df_fire)>0:
            esi_weighted, esi_unweighted = esi_timeseries(df_fire, start_time)
            esi_weighted = pd.concat([esi_weighted, pd.DataFrame({'irwinID':[fireID]*len(esi_weighted)})], axis=1)
            esi_unweighted = pd.concat([esi_unweighted, pd.DataFrame({'irwinID':[fireID]*len(esi_unweighted)})], axis=1)
            print(esi_weighted)
            print(esi_unweighted)
        
        
        esi_weighted.to_csv('./fire_features_esi/'+fireID+str(years[jj])+'_Daily_ESI_Weighted_'+str(start_time)+'Z_day_start.csv') #daily averages
        esi_unweighted.to_csv('./fire_features_esi/'+fireID+str(years[jj])+'_Daily_ESI_Unweighted_'+str(start_time)+'Z_day_start.csv') #daily averages
        
    

