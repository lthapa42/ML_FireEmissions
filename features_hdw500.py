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
def hdw_lagged_timeseries(df,day_start_hour):  #with the wind speed
    varis_hrrr_derived = ['day','hd0w0', 'hd1w0','hd2w0', 'hd3w0']#, 'hd4w0', 'hd5w0',
    df_hdw_weighted = generate_df(varis_hrrr_derived, len(df))
    df_hdw_unweighted = generate_df(varis_hrrr_derived, len(df))
    
    #do the intersection, in parallel
    #print(tic)
    hrrr_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],'HRRR_GRID',3000) 
                                 for ii in range(len(df)))
    
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
        
        times_back = pd.date_range(start=np.datetime64(today)-np.timedelta64(3,'D'), end=np.datetime64(today)+
                                   np.timedelta64(36,'h'),freq='H')
        files_back,times_back_used = make_file_namelist(times_back,'/data2/lthapa/ML_daily/pygraf/processed_hrrr_hdw500/Processed_HRRR_YYYYMMDDHH_HDW500.nc')
        #load in all the merra files associated with this lookback window
        dat_hrrr = xr.open_mfdataset(files_back,concat_dim='time',combine='nested',compat='override', coords='all')
        dat_hrrr = dat_hrrr.assign_coords({'time': times_back_used})
        dat_hrrr = dat_hrrr.resample(time='h').asfreq()
        #print(dat_hrrr['time'].values)
        #define the days
        day0= np.datetime64(today)+np.timedelta64(12,'h')
        day1 =day0-np.timedelta64(1,'D')
        day2 =day0-np.timedelta64(2,'D')
        day3 =day0-np.timedelta64(3,'D')

        #define the times we will select for VPD
        times_0 = pd.date_range(start=day0, end=day0+np.timedelta64(23,'h'),freq='H')
        times_1 = pd.date_range(start=day1, end=day1+np.timedelta64(23,'h'),freq='H')
        times_2 = pd.date_range(start=day2, end=day2+np.timedelta64(23,'h'),freq='H')
        times_3 = pd.date_range(start=day3, end=day3+np.timedelta64(23,'h'),freq='H')
        
        w0 = dat_hrrr['wind_max'].sel(time=times_0, grid_yt = np.unique(intersection_sub['row'].values),grid_xt = np.unique(intersection_sub['col'].values))
        hd0 = dat_hrrr['vpd_max'].sel(time=times_0, grid_yt = np.unique(intersection_sub['row'].values),grid_xt = np.unique(intersection_sub['col'].values))
        hd1 = dat_hrrr['vpd_max'].sel(time=times_1, grid_yt = np.unique(intersection_sub['row'].values),grid_xt = np.unique(intersection_sub['col'].values))
        hd1=hd1.assign_coords({'time':w0['time'].values})
        hd2 = dat_hrrr['vpd_max'].sel(time=times_2, grid_yt = np.unique(intersection_sub['row'].values),grid_xt = np.unique(intersection_sub['col'].values))
        hd2=hd2.assign_coords({'time':w0['time'].values})
        hd3 = dat_hrrr['vpd_max'].sel(time=times_3, grid_yt = np.unique(intersection_sub['row'].values),grid_xt = np.unique(intersection_sub['col'].values))
        hd3=hd3.assign_coords({'time':w0['time'].values})
            
        hd0w0 = hd0*w0
        hd1w0 = hd1*w0
        hd2w0 = hd2*w0
        hd3w0 = hd3*w0

        hd0w0_daily_mean = hd0w0.resample(time='24H',base=day_start_hour, label='left').mean(dim='time') #take the daily mean        
        hd1w0_daily_mean = hd1w0.resample(time='24H',base=day_start_hour, label='left').mean(dim='time') #take the daily mean        
        hd2w0_daily_mean = hd2w0.resample(time='24H',base=day_start_hour, label='left').mean(dim='time') #take the daily mean        
        hd3w0_daily_mean = hd3w0.resample(time='24H',base=day_start_hour, label='left').mean(dim='time') #take the daily mean        

        
        df_hdw_weighted['day'].iloc[count] =today# pd.to_datetime([str(day_start_hour)+ 'Z Start Day'].values[count])
        df_hdw_unweighted['day'].iloc[count] =today# pd.to_datetime([str(day_start_hour)+ 'Z Start Day'].values[count])

        #WEIGHTED
        df_hdw_weighted.loc[count, ('hd0w0')] = np.nansum((hd0w0_daily_mean.values)*(intersection_sub['weights'].values))
        df_hdw_weighted.loc[count, ('hd1w0')] = np.nansum((hd1w0_daily_mean.values)*(intersection_sub['weights'].values))
        df_hdw_weighted.loc[count, ('hd2w0')] = np.nansum((hd2w0_daily_mean.values)*(intersection_sub['weights'].values))
        df_hdw_weighted.loc[count, ('hd3w0')] = np.nansum((hd3w0_daily_mean.values)*(intersection_sub['weights'].values))

        #UNWEIGHTED
        df_hdw_unweighted.loc[count, ('hd0w0')] = np.nanmean((hd0w0_daily_mean.values))
        df_hdw_unweighted.loc[count, ('hd1w0')] = np.nanmean((hd1w0_daily_mean.values))
        df_hdw_unweighted.loc[count, ('hd2w0')] = np.nanmean((hd2w0_daily_mean.values))
        df_hdw_unweighted.loc[count, ('hd3w0')] = np.nanmean((hd3w0_daily_mean.values))
        
        count=count+1
    return df_hdw_weighted, df_hdw_unweighted


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
    #rave_all = pd.DataFrame()
    #hwp_weighted_all = pd.DataFrame()
    #hwp_unweighted_all = pd.DataFrame()
    hdw_weighted_all = pd.DataFrame()
    hdw_unweighted_all = pd.DataFrame()
    for ii in range(len(irwinIDs)):
        print(ii)
        fireID=irwinIDs[ii]
        print(fireID)
        df_fire = fire_daily[fire_daily['irwinID']==fireID] #this is what gets fed to the feature selection code
        
        #maybe we filter 12Z Start dates to where we have data?
        days=np.array(df_fire['12Z Start Day'].values, dtype='datetime64')
        df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                              (days<=np.datetime64(str(years[jj]+1)+'-01-01'))].reset_index(drop=True)
        
        
        #HRRR HDW
        hdw_weighted, hdw_unweighted = hdw_lagged_timeseries(df_fire, start_time)
        hdw_weighted = pd.concat([hdw_weighted, pd.DataFrame({'irwinID':[fireID]*len(hdw_weighted)})], axis=1)
        hdw_unweighted = pd.concat([hdw_unweighted, pd.DataFrame({'irwinID':[fireID]*len(hdw_unweighted)})], axis=1)
        print(hdw_weighted)
        #print(hdw_unweighted)
        
   	 
        hdw_weighted.to_csv('./fire_features_hdw500/'+fireID+str(years[jj])+'_Daily_HDW_Weighted_'+str(start_time)+'Z_day_start.csv') #daily averages
        hdw_unweighted.to_csv('./fire_features_hdw500/'+fireID+str(years[jj])+'_Daily_HDW_Unweighted_'+str(start_time)+'Z_day_start.csv') #daily average        
    
