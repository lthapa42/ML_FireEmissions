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


def slope_timeseries(df, day_start_hour):
    #do the intersection, in parallel
    tic = datetime.datetime.now()
    slope_intersections = Parallel(n_jobs=10)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],'SLOPE_GRID_990M',2000) 
                                 for ii in range(len(df)))
    toc = datetime.datetime.now()
    print(toc-tic)
    print([slope_intersections[jj]['weights'].sum() for jj in range(len(slope_intersections))])

    
    fire_slope_intersection=gpd.GeoDataFrame(pd.concat(slope_intersections, ignore_index=True))
    fire_slope_intersection.set_geometry(col='geometry')
    fire_slope_intersection = fire_slope_intersection.set_index([str(day_start_hour)+ 'Z Start Day', 'row', 'col'])
    
    fire_slope_intersection_xr = fire_slope_intersection.to_xarray()
    
    #nc["cdd_hdd"] = xr.where(nc["tavg"] > 65, nc["tavg"] - 65, 65 - nc["tavg"])
    fire_slope_intersection_xr['weights_mask'] = xr.where(fire_slope_intersection_xr['weights']>0,1, np.nan)
    
    #load in SLOPE data associated with the fire (it's only one dataset)  
    #open the SLOPE files
    path_slope = '/data2/lthapa/ML_daily/slope_990m_LF2020.nc'
    dat_slope = xr.open_dataset(path_slope) #map is fixed in time
    #print(dat_slope)
    
    dat_slope = dat_slope.assign_coords({'time': pd.to_datetime(fire_slope_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)})
    data_slope_mean = dat_slope['mean_slope'].expand_dims({'time': pd.to_datetime(fire_slope_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)}) #the mean slope expanded over all the days
    data_slope_std = dat_slope['std_slope'].expand_dims({'time': pd.to_datetime(fire_slope_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)})
    
    dat_slope_mean_daily_sub = data_slope_mean.sel(row = fire_slope_intersection_xr['row'].values, 
                                          col = fire_slope_intersection_xr['col'].values,
                      time = pd.to_datetime(fire_slope_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values), method='nearest')
    
    dat_slope_std_daily_sub = data_slope_std.sel(row = fire_slope_intersection_xr['row'].values, 
                                          col = fire_slope_intersection_xr['col'].values,
                      time = pd.to_datetime(fire_slope_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values), method='nearest')
    
    ndays = len(fire_slope_intersection_xr[str(day_start_hour)+ 'Z Start Day'])
    
    #preallocate space for the output
    df_slope_weighted = pd.DataFrame({'day':np.zeros(ndays),'MEAN_SLOPE':np.zeros(ndays),'STD_SLOPE':np.zeros(ndays)})
    df_slope_unweighted = pd.DataFrame({'day':np.zeros(ndays),'MEAN_SLOPE':np.zeros(ndays),'STD_SLOPE':np.zeros(ndays)})


    df_slope_weighted['day'].iloc[:] = pd.to_datetime(fire_slope_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)
    df_slope_unweighted['day'].iloc[:] = pd.to_datetime(fire_slope_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)

    #mean slope
    df_slope_weighted['MEAN_SLOPE'] = np.nansum(fire_slope_intersection_xr['weights'].values*dat_slope_mean_daily_sub.values, axis=(1,2))
    df_slope_unweighted['MEAN_SLOPE'] = np.nanmean(fire_slope_intersection_xr['weights_mask'].values*dat_slope_mean_daily_sub.values, axis=(1,2))
    
    
    #std slope
    df_slope_weighted['STD_SLOPE'] = np.nansum(fire_slope_intersection_xr['weights'].values*dat_slope_std_daily_sub.values, axis=(1,2))
    df_slope_unweighted['STD_SLOPE'] = np.nanmean(fire_slope_intersection_xr['weights_mask'].values*dat_slope_std_daily_sub.values, axis=(1,2))
    return df_slope_weighted, df_slope_unweighted
    

path_poly = '/data2/lthapa/ML_daily/fire_polygons/'
suffix_poly = 'Z_Day_Start.geojson'
start_time=12

#years =[2019,2020,2021]
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
    

    
    for ii in range(len(irwinIDs)):
        print(ii)
        fireID=irwinIDs[ii]
        print(fireID)
        df_fire = fire_daily[fire_daily['irwinID']==fireID] #this is what gets fed to the feature selection code
        
        #maybe we filter 12Z Start dates to where we have data?
        days=np.array(df_fire['12Z Start Day'].values, dtype='datetime64')
        df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                              (days<=np.datetime64(str(years[jj]+1)+'-01-01'))].reset_index(drop=True)

        #SLOPE
        slope_weighted, slope_unweighted = slope_timeseries(df_fire, start_time)
        slope_weighted = pd.concat([slope_weighted, pd.DataFrame({'irwinID':[fireID]*len(slope_weighted)})], axis=1)
        slope_unweighted = pd.concat([slope_unweighted, pd.DataFrame({'irwinID':[fireID]*len(slope_unweighted)})], axis=1)
        print(slope_weighted)
        print(slope_unweighted)
        
    
        slope_weighted.to_csv('./fire_features_slope/'+fireID+str(years[jj])+'_Daily_SLOPE_Weighted_'+str(start_time)+'Z_day_start.csv') #daily averages
        slope_unweighted.to_csv('./fire_features_slope/'+fireID+str(years[jj])+'_Daily_SLOPE_Unweighted_'+str(start_time)+'Z_day_start.csv') #daily averages