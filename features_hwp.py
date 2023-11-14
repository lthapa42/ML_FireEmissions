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

#HRRR_WS formulation from, take time mean, then take weighted average. For HDW, multiply the weighted means of VPD and WIND
def hwp_timeseries(df,day_start_hour):  #with the wind speed
    varis_hrrr_derived = ['day','hwp','hwp_1'] #'hd1w0','hd2w0', 'hd3w0', 'hd4w0', 'hd5w0',
    
    #return both!
    df_hwp_weighted = generate_df(varis_hrrr_derived, len(df))
    df_hwp_unweighted = generate_df(varis_hrrr_derived, len(df))

    #do the intersection, in parallel
    #print(tic)
    hrrr_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],'HRRR_GRID',3000) 
                                 for ii in range(len(df)))
    print([hrrr_intersections[jj]['weights'].sum() for jj in range(len(hrrr_intersections))])

    fire_hrrr_intersection=gpd.GeoDataFrame(pd.concat(hrrr_intersections, ignore_index=True))
    fire_hrrr_intersection.set_geometry(col='geometry')
    
    
    #loop over all of the days we have intersections
    times_intersect = np.unique(fire_hrrr_intersection[str(day_start_hour)+ 'Z Start Day'].values)
    times_utc = np.unique(fire_hrrr_intersection['UTC Day'].values)
    
    count = 0
    for today in times_intersect:
        print(today)
        #get the time
        df_sub = fire_hrrr_intersection.iloc[np.where(fire_hrrr_intersection[str(day_start_hour)+ 'Z Start Day'].values==today)]
        df_sub = df_sub.set_index([str(day_start_hour)+ 'Z Start Day', 'row', 'col'])
        df_sub=df_sub[~df_sub.index.duplicated()]
        intersection_sub = df_sub.to_xarray() #polygon and weights for today
        intersection_sub['weights_mask'] =xr.where(intersection_sub['weights']>0,1, np.nan)
        
        times_back = pd.date_range(start=np.datetime64(today)-np.timedelta64(1,'D'), end=np.datetime64(today)+
                                   np.timedelta64(1,'D'),freq='H')
        #print(times_back)
        files_back,times_back_used = make_file_namelist(times_back,'/data2/lthapa/ML_daily/pygraf/processed_hrrr_hdw_hwp/Processed_HRRR_YYYYMMDDHH_HDW_HWP.nc')
        #load in all the merra files associated with this lookback window
        dat_hrrr = xr.open_mfdataset(files_back,concat_dim='time',combine='nested',compat='override', coords='all')
        dat_hrrr = dat_hrrr.assign_coords({'time': times_back_used})
        
        #dat_hrrr['hd0w0'] = dat_hrrr['wind_speed']*dat_hrrr['vpd_2m']
        #print(dat_hrrr)
        
        hrrr_daily_mean = dat_hrrr.resample(time='24H',base=day_start_hour, label='left').mean(dim='time') #take the daily mean        
        
        hrrr_daily_mean_region = hrrr_daily_mean.sel(grid_yt = np.unique(intersection_sub['row'].values),
                                                    grid_xt = np.unique(intersection_sub['col'].values)) #get the location of the overlaps
        hrrr_daily_mean_region = hrrr_daily_mean_region.where(hrrr_daily_mean_region['hwp']!=0) #mask out zeroes
        print(hrrr_daily_mean_region['hwp'])
        df_hwp_weighted['day'].iloc[count] =today# pd.to_datetime([str(day_start_hour)+ 'Z Start Day'].values[count])            df_hrrr_derived.loc[count, ('hd0w0')] = np.nansum((hrrr_daily_mean_region['hd0w0'].sel(time=np.datetime64(today+ ' '+str(day_start_hour)+':00:00')).values)*(intersection_sub['weights'].values))
        df_hwp_unweighted['day'].iloc[count] =today# pd.to_datetime([str(day_start_hour)+ 'Z Start Day'].values[count])            df_hrrr_derived.loc[count, ('hd0w0')] = np.nansum((hrrr_daily_mean_region['hd0w0'].sel(time=np.datetime64(today+ ' '+str(day_start_hour)+':00:00')).values)*(intersection_sub['weights'].values))

        df_hwp_weighted.loc[count, ('hwp')] =np.nansum((hrrr_daily_mean_region['hwp'].sel(time=np.datetime64(today+ ' '+str(day_start_hour)+':00:00')).values)*(intersection_sub['weights'].values))  
        df_hwp_weighted.loc[count, ('hwp_1')] =np.nansum((hrrr_daily_mean_region['hwp'].sel(time=np.datetime64(today)+ np.timedelta64(12,'h')-np.timedelta64(1,'D')).values)*(intersection_sub['weights'].values))  
        df_hwp_unweighted.loc[count, ('hwp')] = np.nanmean((hrrr_daily_mean_region['hwp'].sel(time=np.datetime64(today+ ' '+str(day_start_hour)+':00:00')).values)*(intersection_sub['weights_mask'].values))

        dat_hrrr.close()
        count =count+1
        #print(df_hrrr_derived)
    return df_hwp_weighted, df_hwp_unweighted

path_poly = '/data2/lthapa/ML_daily/fire_polygons/'
suffix_poly = 'Z_Day_Start.geojson'
start_time=12

#years =[2019,2020,2021]
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
    #irwinIDs = ['79E39F05-5CF7-49C1-B211-11B850C2C643']
    print('We are processing ' +str(len(irwinIDs)) + ' unique fires for '+ str(years[jj]))
    
    hwp_weighted_all = pd.DataFrame()
    hwp_unweighted_all = pd.DataFrame()
    for ii in range(len(irwinIDs)):
        print(ii)
        fireID=irwinIDs[ii]
        print(fireID)
        df_fire = fire_daily[fire_daily['irwinID']==fireID] #this is what gets fed to the feature selection code
        
        #maybe we filter 12Z Start dates to where we have data?
        days=np.array(df_fire['12Z Start Day'].values, dtype='datetime64')
        df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                              (days<=np.datetime64(str(years[jj]+1)+'-01-01'))].reset_index(drop=True)
        #HRRR HWP
        hwp_weighted, hwp_unweighted = hwp_timeseries(df_fire, start_time)
        
        
        hwp_weighted = pd.concat([hwp_weighted, pd.DataFrame({'irwinID':[fireID]*len(hwp_weighted)})], axis=1)
        hwp_unweighted = pd.concat([hwp_unweighted, pd.DataFrame({'irwinID':[fireID]*len(hwp_unweighted)})], axis=1)

        print(hwp_weighted)
        print(hwp_unweighted)
        
        hwp_weighted.to_csv('./fire_features_hwp/'+fireID+str(years[jj])+'_Daily_HWP_Weighted_'+str(start_time)+'Z_day_start.csv') #daily averages
        hwp_unweighted.to_csv('./fire_features_hwp/'+fireID+str(years[jj])+'_Daily_HWP_Unweighted_'+str(start_time)+'Z_day_start.csv') #daily averages      

    
        


        

