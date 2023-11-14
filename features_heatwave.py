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

def count_high_heatwave_days(fire_intersection, time_gridmet_file, heatwave_days):
    
    #load in the gridmet file to check for exceedances
    gridmet_filename = '/data2/lthapa/'+time_gridmet_file[0:4]+\
                        '/GRIDMET/gridmet_all_'+time_gridmet_file+'.nc'
    gridmet_today = xr.open_dataset(gridmet_filename)
    
    gridmet_today_sub = gridmet_today.sel(lat=fire_intersection['lat'].values, lon=fire_intersection['lon'].values)
    
    heatwave_in_polygon = (gridmet_today_sub['is_high_heatwave']==1) & (fire_intersection['weights_mask']==1)

    if heatwave_in_polygon.any():
        heatwave_days_inc = heatwave_days+1
        time_gridmet_file_back = str(np.datetime64(time_gridmet_file)-np.timedelta64(1,'D'))
        return count_high_heatwave_days(fire_intersection, time_gridmet_file_back,heatwave_days_inc)
    
    else:
        heatwave_days_inc=heatwave_days
        return heatwave_days_inc
    
def count_highlow_heatwave_days(fire_intersection, time_gridmet_file, heatwave_days):
    
    #load in the gridmet file to check for exceedances
    gridmet_filename = '/data2/lthapa/'+time_gridmet_file[0:4]+\
                        '/GRIDMET/gridmet_all_'+time_gridmet_file+'.nc'
    gridmet_today = xr.open_dataset(gridmet_filename)
    
    gridmet_today_sub = gridmet_today.sel(lat=fire_intersection['lat'].values, lon=fire_intersection['lon'].values)
    
    heatwave_in_polygon = (gridmet_today_sub['is_highlow_heatwave']==1) & (fire_intersection['weights_mask']==1)

    if heatwave_in_polygon.any():
        heatwave_days_inc = heatwave_days+1
        time_gridmet_file_back = str(np.datetime64(time_gridmet_file)-np.timedelta64(1,'D'))
        return count_highlow_heatwave_days(fire_intersection, time_gridmet_file_back,heatwave_days_inc)
    
    else:
        heatwave_days_inc=heatwave_days
        return heatwave_days_inc
    
def count_heatwave_days(fire_intersection, time_gridmet_file, heatwave_days, var_to_check):
    
    #load in the gridmet file to check for exceedances
    gridmet_filename = '/data2/lthapa/'+time_gridmet_file[0:4]+\
                        '/GRIDMET/gridmet_all_'+time_gridmet_file+'.nc'
    gridmet_today = xr.open_dataset(gridmet_filename)
    
    gridmet_today_sub = gridmet_today.sel(lat=fire_intersection['lat'].values, lon=fire_intersection['lon'].values)
    
    heatwave_in_polygon = (gridmet_today_sub[var_to_check]==1) & (fire_intersection['weights_mask']==1)

    if heatwave_in_polygon.any():
        heatwave_days_inc = heatwave_days+1
        time_gridmet_file_back = str(np.datetime64(time_gridmet_file)-np.timedelta64(1,'D'))
        return count_heatwave_days(fire_intersection, time_gridmet_file_back,heatwave_days_inc,var_to_check)
    
    else:
        heatwave_days_inc=heatwave_days
        return heatwave_days_inc
    

def heatwave_timeseries(df, day_start_hour):
    varis = ['day','days_in_high_heatwave', 'days_in_highlow_heatwave'] 
    df_heatwave= generate_df(varis, len(df))
    
    #do the intersection, in parallel
    gridmet_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],'WESTUS_GRIDMET_GRID',100000) 
                                 for ii in range(len(df)))
    print([gridmet_intersections[jj]['weights'].sum() for jj in range(len(gridmet_intersections))])

    fire_gridmet_intersection=gpd.GeoDataFrame(pd.concat(gridmet_intersections, ignore_index=True))
    fire_gridmet_intersection.set_geometry(col='geometry')    
    fire_gridmet_intersection = fire_gridmet_intersection.set_index([str(day_start_hour)+ 'Z Start Day', 'lat', 'lon'])

    fire_gridmet_intersection=fire_gridmet_intersection[~fire_gridmet_intersection.index.duplicated()]
    
    fire_gridmet_intersection_xr = fire_gridmet_intersection.to_xarray()
    fire_gridmet_intersection_xr['weights_mask'] = xr.where(fire_gridmet_intersection_xr['weights']>0,1, np.nan)
    
    fire_gridmet_intersection_xr = fire_gridmet_intersection_xr.rename(name_dict = {'12Z Start Day': 'Start_Day'})
    
    print(fire_gridmet_intersection_xr)
    #print(fire_gridmet_intersection_xr['12Z Start Day'])
    
    for xx in range(len(fire_gridmet_intersection_xr['Start_Day'].values)): #loop over all the days where we have polygons
        poly_time = fire_gridmet_intersection_xr['Start_Day'].values[xx]
        print(poly_time,type(poly_time))
        
        intersection_today = fire_gridmet_intersection_xr.sel(Start_Day=poly_time)
        #print(intersection_today)
        
        days_in_high_heatwave = 0 #start by assuming we are out of the heatwave
        df_heatwave['days_in_high_heatwave'].iloc[xx] = count_heatwave_days(intersection_today, poly_time, days_in_high_heatwave, 'is_high_heatwave')
        
        days_in_highlow_heatwave = 0 #start by assuming we are out of the heatwave
        df_heatwave['days_in_highlow_heatwave'].iloc[xx] = count_heatwave_days(intersection_today, poly_time, days_in_highlow_heatwave, 'is_highlow_heatwave')
        
    df_heatwave['day'].iloc[:] = pd.to_datetime(fire_gridmet_intersection_xr['Start_Day'].values)
    
    return df_heatwave

path_poly = '/data2/lthapa/ML_daily/fire_polygons/'
suffix_poly = 'Z_Day_Start.geojson'
start_time=12

years =[2019,2020,2021]
#years = [2019]



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
    
    for ii in range(len(irwinIDs)):
        print(ii)
        fireID=irwinIDs[ii]
        print(fireID)
        df_fire = fire_daily[fire_daily['irwinID']==fireID] #this is what gets fed to the feature selection code
        
        #maybe we filter 12Z Start dates to where we have data?
        days=np.array(df_fire['12Z Start Day'].values, dtype='datetime64')
        df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                             (days<=np.datetime64(str(years[jj]+1)+'-01-01'))].reset_index(drop=True)

        #HEATWAVE
        if len(df_fire)>1:
            heatwave = heatwave_timeseries(df_fire, start_time)
            heatwave = pd.concat([heatwave, pd.DataFrame({'irwinID':[fireID]*len(heatwave)})], axis=1) #add IrwinID
            print(heatwave)

            heatwave.to_csv('./fire_features_heatwave/'+fireID+str(years[jj])+'_Daily_HEATWAVE_'+str(start_time)+'Z_day_start.csv') #daily averages      
    

