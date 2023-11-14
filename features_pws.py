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


def pws_timeseries(df, day_start_hour):
    #do the intersection, in parallel
    tic = datetime.datetime.now()
    pws_intersections = Parallel(n_jobs=10)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],'PWS_GRID',4000) 
                                 for ii in range(len(df)))
    toc = datetime.datetime.now()
    print(toc-tic)
    print([pws_intersections[jj]['weights'].sum() for jj in range(len(pws_intersections))])

    
    fire_pws_intersection=gpd.GeoDataFrame(pd.concat(pws_intersections, ignore_index=True))
    fire_pws_intersection.set_geometry(col='geometry')
    fire_pws_intersection = fire_pws_intersection.set_index([str(day_start_hour)+ 'Z Start Day', 'lat', 'lon'])
    
    fire_pws_intersection_xr = fire_pws_intersection.to_xarray()
    
    #nc["cdd_hdd"] = xr.where(nc["tavg"] > 65, nc["tavg"] - 65, 65 - nc["tavg"])
    fire_pws_intersection_xr['weights_mask'] = xr.where(fire_pws_intersection_xr['weights']>0,1, np.nan)
    
    #load in PWS data associated with the fire (it's only one dataset)  
    #open the PWS files
    path_pws = '/data2/lthapa/PWS_6_jan_2021.nc'
    dat_pws = xr.open_dataset(path_pws) #map is fixed in time
    #print(dat_pws)
    
    dat_pws = dat_pws.assign_coords({'time': pd.to_datetime(fire_pws_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)})
    dat_pws_daily = dat_pws['Band1'].expand_dims({'time': pd.to_datetime(fire_pws_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)}) #the PWS expanded over all the days
    
    dat_pws_daily_sub = dat_pws_daily.sel(lat = fire_pws_intersection_xr['lat'].values, 
                                          lon = fire_pws_intersection_xr['lon'].values,
                      time = pd.to_datetime(fire_pws_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values), method='nearest')
    ndays = len(fire_pws_intersection_xr[str(day_start_hour)+ 'Z Start Day'])
    
    #preallocate space for the output
    df_pws_weighted = pd.DataFrame({'day':np.zeros(ndays),'PWS':np.zeros(ndays)})
    df_pws_unweighted = pd.DataFrame({'day':np.zeros(ndays),'PWS':np.zeros(ndays)})


    df_pws_weighted['day'].iloc[:] = pd.to_datetime(fire_pws_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)
    df_pws_unweighted['day'].iloc[:] = pd.to_datetime(fire_pws_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)

    varis=['PWS']
    for var in varis:
        df_pws_weighted[var] = np.nansum(fire_pws_intersection_xr['weights'].values*dat_pws_daily_sub.values, axis=(1,2)) #WEIGHTED AVERAGE
        df_pws_unweighted[var] = np.nanmean(fire_pws_intersection_xr['weights_mask'].values*dat_pws_daily_sub.values, axis=(1,2)) #MASK AND AVERAGE
        #print(np.nanmean(dat_pws_daily_sub.values, axis=(1,2)))
        #df_pws[var] = dat_pws_daily_sub.mean(dim=['lat','lon'], skipna=True)
    return df_pws_weighted, df_pws_unweighted
    

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
    

    pws_weighted_all = pd.DataFrame() 
    pws_unweighted_all = pd.DataFrame()
    
    for ii in range(len(irwinIDs)):
        print(ii)
        fireID=irwinIDs[ii]
        print(fireID)
        df_fire = fire_daily[fire_daily['irwinID']==fireID] #this is what gets fed to the feature selection code
        
        #maybe we filter 12Z Start dates to where we have data?
        days=np.array(df_fire['12Z Start Day'].values, dtype='datetime64')
        df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                              (days<=np.datetime64(str(years[jj]+1)+'-01-01'))].reset_index(drop=True)

        #PWS
        pws_weighted, pws_unweighted = pws_timeseries(df_fire, start_time)
        pws_weighted = pd.concat([pws_weighted, pd.DataFrame({'irwinID':[fireID]*len(pws_weighted)})], axis=1)
        pws_unweighted = pd.concat([pws_unweighted, pd.DataFrame({'irwinID':[fireID]*len(pws_unweighted)})], axis=1)
        print(pws_weighted)
        print(pws_unweighted)
        
        pws_weighted_all = pd.concat([pws_weighted_all, pws_weighted], axis=0).reset_index(drop=True)
        pws_unweighted_all = pd.concat([pws_unweighted_all, pws_unweighted], axis=0).reset_index(drop=True)
    
    pws_weighted_all.to_csv('./fire_features_3/'+'ClippedFires'+str(years[jj])+'_Daily_PWS_Weighted_'+str(start_time)+'Z_day_start.csv') #daily averages
    pws_unweighted_all.to_csv('./fire_features_3/'+'ClippedFires'+str(years[jj])+'_Daily_PWS_Unweighted_'+str(start_time)+'Z_day_start.csv') #daily averages
