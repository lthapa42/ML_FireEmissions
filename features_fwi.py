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


def imerg_fwi_timeseries(df, day_start_hour):
    varis = ['day','IMERG.FINAL.v6_DC','IMERG.FINAL.v6_DMC','IMERG.FINAL.v6_FFMC',
             'IMERG.FINAL.v6_ISI','IMERG.FINAL.v6_BUI','IMERG.FINAL.v6_FWI',
             'IMERG.FINAL.v6_DSR'] 
    df_imerg_weighted = generate_df(varis, len(df))
    df_imerg_unweighted = generate_df(varis, len(df))
    #do the intersection, in parallel
    fwi_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],'IMERG_FWI_GRID',10000) 
                                 for ii in range(len(df)))
    print([fwi_intersections[jj]['weights'].sum() for jj in range(len(fwi_intersections))])

    fire_fwi_intersection=gpd.GeoDataFrame(pd.concat(fwi_intersections, ignore_index=True))
    fire_fwi_intersection.set_geometry(col='geometry')    
    fire_fwi_intersection = fire_fwi_intersection.set_index([str(day_start_hour)+ 'Z Start Day', 'lat', 'lon'])

    fire_fwi_intersection=fire_fwi_intersection[~fire_fwi_intersection.index.duplicated()]
    
    fire_fwi_intersection_xr = fire_fwi_intersection.to_xarray()
    fire_fwi_intersection_xr['weights_mask'] = xr.where(fire_fwi_intersection_xr['weights']>0,1, np.nan)

    
    #load in FWI data associated with the fire
    times = pd.date_range(np.datetime64(df['UTC Day'].iloc[0]),
                        np.datetime64(df['UTC Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))
    fwi_filenames,times_back_used = make_file_namelist(times,'/data2/lthapa/YYYY/FWI_IMERG/WESTUS_FWI.IMERG.FINAL.v6.Daily.Default.YYYYMMDD.nc')
    
    dat_fwi = xr.open_mfdataset(fwi_filenames,concat_dim='time',combine='nested',compat='override', coords='all')
    dat_fwi = dat_fwi.assign_coords({'time': times_back_used}) #assign coords so we can resample along time
    dat_fwi = dat_fwi.resample(time='1H').pad() #make the data hourly, so we can define the day as 12z-12z instead of 0z-0z
    dat_fwi_mean = dat_fwi.resample(time='24H',base=day_start_hour ,label='left').mean(dim='time') #take the daily mean         
    
    #select the locations and times we want
    dat_fwi_sub = dat_fwi_mean.sel(lat = fire_fwi_intersection_xr['lat'].values, 
                    lon = fire_fwi_intersection_xr['lon'].values).sel(
                    time = pd.to_datetime(fire_fwi_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values+ ' '+
                                         str(day_start_hour)+':00:00'), method='nearest')#these should be lined up correctly


    df_imerg_weighted['day'].iloc[:] = pd.to_datetime(fire_fwi_intersection_xr['12Z Start Day'].values)
    df_imerg_unweighted['day'].iloc[:] = pd.to_datetime(fire_fwi_intersection_xr['12Z Start Day'].values)

    for var in varis[1:len(varis)]:
        df_imerg_weighted[var] = np.nansum(fire_fwi_intersection_xr['weights'].values*dat_fwi_sub[var].values, axis=(1,2)) #weighted average
        df_imerg_unweighted[var] = np.nanmean(fire_fwi_intersection_xr['weights_mask'].values*dat_fwi_sub[var].values,axis=(1,2)) #mask and average
    return df_imerg_weighted, df_imerg_unweighted

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
    imerg_weighted_all = pd.DataFrame()
    imerg_unweighted_all = pd.DataFrame()
    for ii in range(len(irwinIDs)):
        print(ii)
        fireID=irwinIDs[ii]
        print(fireID)
        df_fire = fire_daily[fire_daily['irwinID']==fireID] #this is what gets fed to the feature selection code
        
        #maybe we filter 12Z Start dates to where we have data?
        days=np.array(df_fire['12Z Start Day'].values, dtype='datetime64')
        #df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
        #                      (days<=np.datetime64(str(years[jj]+1)+'-01-01'))].reset_index(drop=True)
        # for 2021
        df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                              (days<=np.datetime64(str(years[jj])+'-10-01'))].reset_index(drop=True)
        #HRRR HDW
        if len(df_fire)>0:
            imerg_weighted, imerg_unweighted = imerg_fwi_timeseries(df_fire, start_time)
            imerg_weighted = pd.concat([imerg_weighted, pd.DataFrame({'irwinID':[fireID]*len(imerg_weighted)})], axis=1)
            imerg_unweighted = pd.concat([imerg_unweighted, pd.DataFrame({'irwinID':[fireID]*len(imerg_unweighted)})], axis=1)
            print(imerg_unweighted)
        
            imerg_weighted_all = pd.concat([imerg_weighted_all, imerg_weighted], axis=0).reset_index(drop=True)
            imerg_unweighted_all = pd.concat([imerg_unweighted_all, imerg_unweighted], axis=0).reset_index(drop=True)
    
    imerg_weighted_all.to_csv('./fire_features_3/'+'ClippedFires'+str(years[jj])+'_Daily_IMERG_FWI_Weighted_'+str(start_time)+'Z_day_start.csv') #daily averages
    imerg_unweighted_all.to_csv('./fire_features_3/'+'ClippedFires'+str(years[jj])+'_Daily_IMERG_FWI_Unweighted_'+str(start_time)+'Z_day_start.csv') #daily average        
    

