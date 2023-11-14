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

def smops_timeseries(df, day_start_hour):
    varis_smops = ['day','Blended_SM']
    df_smops_weighted = generate_df(varis_smops, len(df))
    df_smops_unweighted = generate_df(varis_smops, len(df))

    #do the intersection, in parallel
    smops_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],'SMOPS_GRID',25000) 
                                 for ii in range(len(df)))
    print([smops_intersections[jj]['weights'].sum() for jj in range(len(smops_intersections))])

    fire_smops_intersection=gpd.GeoDataFrame(pd.concat(smops_intersections, ignore_index=True))
    fire_smops_intersection.set_geometry(col='geometry')  
    fire_smops_intersection = fire_smops_intersection.set_index([str(day_start_hour)+'Z Start Day', 'row', 'col'])
    fire_smops_intersection=fire_smops_intersection[~fire_smops_intersection.index.duplicated()]

    fire_smops_intersection_xr = fire_smops_intersection.to_xarray()
    fire_smops_intersection_xr['weights_mask'] = xr.where(fire_smops_intersection_xr['weights']>0,1, np.nan)
    #print(fire_smops_intersection_xr['weights_mask'])

    #load in rave data associated with the fire
    times = pd.date_range(np.datetime64(df[str(day_start_hour)+ 'Z Start Day'].iloc[0]),
                        np.datetime64(df[str(day_start_hour)+ 'Z Start Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))
    smops_filenames,times_back_used = make_file_namelist(times,'/data2/lthapa/YYYY/SMOPS/NPR_SMOPS_CMAP_DYYYYMMDD.nc')

    dat_smops = xr.open_mfdataset(smops_filenames,concat_dim='Time',combine='nested',compat='override', coords='all')
    dat_smops = dat_smops.assign_coords({'Time': times_back_used}) #assign coords so we can select in time
    #print(dat_smops['Time'].values)
    #print(fire_smops_intersection_xr[str(day_start_hour)+'Z Start Day'].values)
    #select the locations and times we want
    dat_smops_sub = dat_smops.isel(Latitude = fire_smops_intersection_xr['row'].values.astype(int), 
                    Longitude = fire_smops_intersection_xr['col'].values.astype(int)).sel(
                    Time = pd.to_datetime(fire_smops_intersection_xr[str(day_start_hour)+'Z Start Day'].values))#these should be lined up correctly

    df_smops_weighted['day'].iloc[:] = pd.to_datetime(fire_smops_intersection_xr[str(day_start_hour)+'Z Start Day'].values)
    df_smops_unweighted['day'].iloc[:] = pd.to_datetime(fire_smops_intersection_xr[str(day_start_hour)+'Z Start Day'].values)

    for var in varis_smops[1:]:
        dat_smops_sub[var]=dat_smops_sub[var].where(dat_smops_sub[var] != -0.0999) #mask out the ocean, there is no soil moisture here
        df_smops_weighted[var] = np.nansum(fire_smops_intersection_xr['weights'].values*dat_smops_sub[var].values, axis=(1,2))
        df_smops_unweighted[var] = np.nanmean(fire_smops_intersection_xr['weights_mask'].values*dat_smops_sub[var].values, axis=(1,2)) #MASK AND AVERAGE
    
    return df_smops_weighted, df_smops_unweighted


# main method
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
        if len(df_fire)>0:
            smops_weighted, smops_unweighted = smops_timeseries(df_fire, start_time)
            smops_weighted = pd.concat([smops_weighted, pd.DataFrame({'irwinID':[fireID]*len(smops_weighted)})], axis=1)
            smops_unweighted = pd.concat([smops_unweighted, pd.DataFrame({'irwinID':[fireID]*len(smops_unweighted)})], axis=1)
            print(smops_unweighted)
        
            smops_weighted_all = pd.concat([smops_weighted_all, smops_weighted], axis=0).reset_index(drop=True)
            smops_unweighted_all = pd.concat([smops_unweighted_all, smops_unweighted], axis=0).reset_index(drop=True)
    
    smops_weighted_all.to_csv('./fire_features_3/'+'ClippedFires'+str(years[jj])+'_Daily_SMOPS_Weighted_'+str(start_time)+'Z_day_start.csv') #daily averages
    smops_unweighted_all.to_csv('./fire_features_3/'+'ClippedFires'+str(years[jj])+'_Daily_SMOPS_Unweighted_'+str(start_time)+'Z_day_start.csv') #daily average  
