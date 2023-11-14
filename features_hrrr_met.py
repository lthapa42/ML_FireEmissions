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

def hrrrmet_timeseries(df, day_start_hour):
    varis_hrrrmet = ['day','wind_speed', 'vpd_2m']
    
    df_hrrrmet_weighted = generate_df(varis_hrrrmet, len(df))
    df_hrrrmet_unweighted = generate_df(varis_hrrrmet, len(df))

    #do the intersection, in parallel
    hrrrmet_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],'HRRR_GRID',3000) 
                                 for ii in range(len(df)))
    print([hrrrmet_intersections[jj]['weights'].sum() for jj in range(len(hrrrmet_intersections))])

    fire_hrrrmet_intersection=gpd.GeoDataFrame(pd.concat(hrrrmet_intersections, ignore_index=True))
    fire_hrrrmet_intersection.set_geometry(col='geometry')  
    fire_hrrrmet_intersection = fire_hrrrmet_intersection.set_index([str(day_start_hour)+'Z Start Day', 'row', 'col'])
    fire_hrrrmet_intersection=fire_hrrrmet_intersection[~fire_hrrrmet_intersection.index.duplicated()]

    fire_hrrrmet_intersection_xr = fire_hrrrmet_intersection.to_xarray()
    fire_hrrrmet_intersection_xr['weights_mask'] = xr.where(fire_hrrrmet_intersection_xr['weights']>0,1, np.nan)
    #print(fire_smops_intersection_xr['weights_mask'])

    #load in rave data associated with the fire
    times = pd.date_range(np.datetime64(df[str(day_start_hour)+ 'Z Start Day'].iloc[0]),
                        np.datetime64(df[str(day_start_hour)+ 'Z Start Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))
    hrrrmet_filenames,times_back_used = make_file_namelist(times,'/data2/lthapa/ML_daily/pygraf/processed_hrrr_hdw_hwp/Processed_HRRR_YYYYMMDDHH_HDW_HWP.nc')

    dat_hrrrmet = xr.open_mfdataset(hrrrmet_filenames,concat_dim='Time',combine='nested',compat='override', coords='all')
    dat_hrrrmet = dat_hrrrmet.assign_coords({'Time': times_back_used}) #assign coords so we can select in time
    #print(dat_smops['Time'].values)
    #print(fire_smops_intersection_xr[str(day_start_hour)+'Z Start Day'].values)
    #select the locations and times we want
    hrrr_daily_mean = dat_hrrrmet.resample(Time='24H',base=day_start_hour, label='left').mean(dim='Time') #take the daily mean       
    hrrr_daily_mean_region = hrrr_daily_mean.sel(grid_yt = np.unique(fire_hrrrmet_intersection_xr['row'].values),
                                                    grid_xt = np.unique(fire_hrrrmet_intersection_xr['col'].values)).sel(
                    Time = pd.to_datetime(fire_hrrrmet_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values+ ' '+
                                         str(day_start_hour)+':00:00'), method='nearest')#these should be lined up correctly

    df_hrrrmet_weighted['day'].iloc[:] = pd.to_datetime(fire_hrrrmet_intersection_xr[str(day_start_hour)+'Z Start Day'].values)
    df_hrrrmet_unweighted['day'].iloc[:] = pd.to_datetime(fire_hrrrmet_intersection_xr[str(day_start_hour)+'Z Start Day'].values)

    for var in varis_hrrrmet[1:]:
        df_hrrrmet_weighted[var] =np.nansum((hrrr_daily_mean_region[var].values)*(fire_hrrrmet_intersection_xr['weights'].values),axis=(1,2))   
        df_hrrrmet_unweighted[var] = np.nanmean((hrrr_daily_mean_region[var].values)*(fire_hrrrmet_intersection_xr['weights_mask'].values),axis=(1,2))
    return df_hrrrmet_weighted, df_hrrrmet_unweighted


# main method
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
    
    smops_weighted_all = pd.DataFrame()
    smops_unweighted_all = pd.DataFrame()
    for ii in range(len(irwinIDs)):
        print(ii)
        fireID=irwinIDs[ii]
        print(fireID)
        df_fire = fire_daily[fire_daily['irwinID']==fireID] #this is what gets fed to the feature selection code
        
        #maybe we filter 12Z Start dates to where we have data?
        days=np.array(df_fire['12Z Start Day'].values, dtype='datetime64')
        df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                              (days<=np.datetime64(str(years[jj]+1)+'-01-01'))].reset_index(drop=True)
        # for 2021
        #df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
        #                      (days<=np.datetime64(str(years[jj])+'-10-01'))].reset_index(drop=True)
        #HRRR HDW
        if len(df_fire)>1:
            hrrrmet_weighted, hrrrmet_unweighted = hrrrmet_timeseries(df_fire, start_time)
            hrrrmet_weighted = pd.concat([hrrrmet_weighted, pd.DataFrame({'irwinID':[fireID]*len(hrrrmet_weighted)})], axis=1)
            hrrrmet_unweighted = pd.concat([hrrrmet_unweighted, pd.DataFrame({'irwinID':[fireID]*len(hrrrmet_unweighted)})], axis=1)
            print(hrrrmet_unweighted)
            
            hrrrmet_weighted.to_csv('./fire_features_hrrrmet/'+fireID+str(years[jj])+'_Daily_HRRRMET_Weighted_'+str(start_time)+'Z_day_start.csv') #daily averages
            hrrrmet_unweighted.to_csv('./fire_features_hrrrmet/'+fireID+str(years[jj])+'_Daily_HRRRMET_Unweighted_'+str(start_time)+'Z_day_start.csv') #daily averages
        
