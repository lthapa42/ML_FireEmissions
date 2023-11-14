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

def fuel_loading_timeseries(df, day_start_hour):
    #do the intersection, in parallel
    tic = datetime.datetime.now()
    fuel_fwi_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],'FUEL_FWI_GRID_990M',2000) 
                                 for ii in range(len(df)))
    toc = datetime.datetime.now()
    print(toc-tic)
    print([fuel_fwi_intersections[jj]['weights'].sum() for jj in range(len(fuel_fwi_intersections))])
    
    fire_fuel_fwi_intersection=gpd.GeoDataFrame(pd.concat(fuel_fwi_intersections, ignore_index=True))
    fire_fuel_fwi_intersection = fire_fuel_fwi_intersection.drop(columns='geometry')
    fire_fuel_fwi_intersection = fire_fuel_fwi_intersection.set_index(['12Z Start Day','row', 'col'])
    fire_fuel_fwi_intersection_xr = fire_fuel_fwi_intersection.to_xarray()
    fire_fuel_fwi_intersection_xr['weights_mask'] = xr.where(fire_fuel_fwi_intersection_xr['weights']>0,1, np.nan)

    path_fuel_fwi = '/data2/lthapa/ML_daily/fuel_fwi_990m.nc'
    dat_fuel_fwi = xr.open_dataset(path_fuel_fwi) #map is fixed in time
    dat_fuel_fwi = dat_fuel_fwi.where(dat_fuel_fwi!=0)
    dat_fuel_fwi_daily = dat_fuel_fwi.expand_dims({'time': pd.to_datetime(fire_fuel_fwi_intersection_xr['12Z Start Day'].values)}) #the PWS expanded over all the days

    dat_fuel_fwi_sub_daily = dat_fuel_fwi_daily.sel(row = fire_fuel_fwi_intersection_xr['row'].values, 
                                        col = fire_fuel_fwi_intersection_xr['col'].values, method='nearest')
    print(dat_fuel_fwi_sub_daily['Extreme_N'])
    #print(fire_fuel_fwi_intersection_xr['weights_mask'].values)
    #preallocate space for the output
    varis = ['day','Extreme_N', 'VeryHigh_N','High_N', 'Moderate_N', 'Low_N']
    df_loading_weighted = generate_df(varis, len(df))
    df_loading_unweighted = generate_df(varis, len(df))

    df_loading_weighted['day'] = df['12Z Start Day'].values
    df_loading_unweighted['day'] = df['12Z Start Day'].values

    for var in varis[1:len(varis)]:

        df_loading_weighted[var] = np.nansum(fire_fuel_fwi_intersection_xr['weights'].values*dat_fuel_fwi_sub_daily[var].values, axis=(1,2))
        df_loading_unweighted[var] = np.nanmean(fire_fuel_fwi_intersection_xr['weights_mask'].values*dat_fuel_fwi_sub_daily[var].values, axis=(1,2))

    return df_loading_weighted, df_loading_unweighted
    
path_poly = '/data2/lthapa/ML_daily/fire_polygons/'
suffix_poly = 'Z_Day_Start.geojson'
start_time=12

#years =[2019,2020,2021]
#years=[2019]
years = [2020,2021]



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
    

    loading_weighted_all = pd.DataFrame() 
    loading_unweighted_all = pd.DataFrame()
    
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
        loading_weighted, loading_unweighted = fuel_loading_timeseries(df_fire, start_time)
        loading_weighted = pd.concat([loading_weighted, pd.DataFrame({'irwinID':[fireID]*len(loading_weighted)})], axis=1)
        loading_unweighted = pd.concat([loading_unweighted, pd.DataFrame({'irwinID':[fireID]*len(loading_unweighted)})], axis=1)
        print(loading_weighted)
    
        loading_weighted.to_csv('./fire_features_loading/'+fireID+str(years[jj])+'_Daily_LOADING_Weighted_'+str(start_time)+'Z_day_start.csv') #daily averages
        loading_unweighted.to_csv('./fire_features_loading/'+fireID+str(years[jj])+'_Daily_LOADING_Unweighted_'+str(start_time)+'Z_day_start.csv') #daily averages
