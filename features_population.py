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


def pop_timeseries(df, day_start_hour):
    #do the intersection, in parallel
    tic = datetime.datetime.now()
    pop_intersections = Parallel(n_jobs=10)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],'POP_GRID',5000) 
                                 for ii in range(len(df)))
    toc = datetime.datetime.now()
    print(toc-tic)
    print([pop_intersections[jj]['weights'].sum() for jj in range(len(pop_intersections))])

    
    fire_pop_intersection=gpd.GeoDataFrame(pd.concat(pop_intersections, ignore_index=True))
    fire_pop_intersection.set_geometry(col='geometry')
    fire_pop_intersection = fire_pop_intersection.set_index([str(day_start_hour)+ 'Z Start Day', 'lat', 'lon'])
    
    fire_pop_intersection_xr = fire_pop_intersection.to_xarray()
    
    #nc["cdd_hdd"] = xr.where(nc["tavg"] > 65, nc["tavg"] - 65, 65 - nc["tavg"])
    fire_pop_intersection_xr['weights_mask'] = xr.where(fire_pop_intersection_xr['weights']>0,1, np.nan)
    
    #load in Pop data associated with the fire (it's only one dataset)  
    #open the Pop files
    path_pop = '/data2/lthapa/static_maps/gpw_v4_population_density_rev11_2pt5_min.nc'
    dat_pop = xr.open_dataset(path_pop) #map is fixed in time
    
    dat_pop = dat_pop.assign_coords({'time': pd.to_datetime(fire_pop_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)})
    dat_pop_daily =  dat_pop['Population Density, v4.11 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes'].sel(raster=5).expand_dims({'time': pd.to_datetime(fire_pop_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)}) #the POP expanded over all the days
    
    dat_pop_daily_sub = dat_pop_daily.sel(latitude = fire_pop_intersection_xr['lat'].values, 
                                          longitude = fire_pop_intersection_xr['lon'].values,
                      time = pd.to_datetime(fire_pop_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values), method='nearest')
    ndays = len(fire_pop_intersection_xr[str(day_start_hour)+ 'Z Start Day'])
    
    #preallocate space for the output
    df_pop_weighted = pd.DataFrame({'day':np.zeros(ndays),'POP_DENSITY':np.zeros(ndays)})
    df_pop_unweighted = pd.DataFrame({'day':np.zeros(ndays),'POP_DENSITY':np.zeros(ndays)})


    df_pop_weighted['day'].iloc[:] = pd.to_datetime(fire_pop_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)
    df_pop_unweighted['day'].iloc[:] = pd.to_datetime(fire_pop_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values)

    varis=['POP_DENSITY']
    for var in varis:
        df_pop_weighted[var] = np.nansum(fire_pop_intersection_xr['weights'].values*dat_pop_daily_sub.values, axis=(1,2)) #WEIGHTED AVERAGE
        df_pop_unweighted[var] = np.nanmean(fire_pop_intersection_xr['weights_mask'].values*dat_pop_daily_sub.values, axis=(1,2)) #MASK AND AVERAGE
        #print(np.nanmean(dat_pws_daily_sub.values, axis=(1,2)))
        #df_pws[var] = dat_pws_daily_sub.mean(dim=['lat','lon'], skipna=True)
    return df_pop_weighted, df_pop_unweighted
    

path_poly = '/data2/lthapa/ML_daily/fire_polygons/'
suffix_poly = 'Z_Day_Start.geojson'
start_time=12

#years =[2020, 2021]
years=[2021]
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
    

    pop_weighted_all = pd.DataFrame() 
    pop_unweighted_all = pd.DataFrame()
    
    for ii in range(len(irwinIDs)):
        print(ii)
        fireID=irwinIDs[ii]
        print(fireID)
        df_fire = fire_daily[fire_daily['irwinID']==fireID] #this is what gets fed to the feature selection code
        
        #maybe we filter 12Z Start dates to where we have data?
        days=np.array(df_fire['12Z Start Day'].values, dtype='datetime64')
        df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                              (days<=np.datetime64(str(years[jj]+1)+'-01-01'))].reset_index(drop=True)

        #POP
        pop_weighted, pop_unweighted = pop_timeseries(df_fire, start_time)
        pop_weighted = pd.concat([pop_weighted, pd.DataFrame({'irwinID':[fireID]*len(pop_weighted)})], axis=1)
        pop_unweighted = pd.concat([pop_unweighted, pd.DataFrame({'irwinID':[fireID]*len(pop_unweighted)})], axis=1)
        print(pop_weighted)
        print(pop_unweighted)
        
        pop_weighted_all = pd.concat([pop_weighted_all, pop_weighted], axis=0).reset_index(drop=True)
        pop_unweighted_all = pd.concat([pop_unweighted_all, pop_unweighted], axis=0).reset_index(drop=True)
    
        pop_weighted.to_csv('./fire_features_pop/'+fireID+str(years[jj])+'_Daily_POP_Weighted_'+str(start_time)+'Z_day_start.csv') #daily averages
        pop_unweighted.to_csv('./fire_features_pop/'+fireID+str(years[jj])+'_Daily_POP_Unweighted_'+str(start_time)+'Z_day_start.csv') #daily averages
