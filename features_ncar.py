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


#ncar grid was 1000m = 1km = grid resolution
# fwi grid was 10000m=10km = grid resolution
def ncar_timeseries(df, day_start_hour):
    varis_ncar = ['day','FMCG2D','FMCGLH2D']
    df_ncar_weighted = generate_df(varis_ncar, len(df))
    df_ncar_unweighted = generate_df(varis_ncar, len(df))

    #do the intersection, in parallel
    ncar_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
                                 (df.iloc[ii:ii+1],'NCAR_MOISTURE_GRID',1000) 
                                 for ii in range(len(df)))
    print([ncar_intersections[jj]['weights'].sum() for jj in range(len(ncar_intersections))])

    
    fire_ncar_intersection=gpd.GeoDataFrame(pd.concat(ncar_intersections, ignore_index=True))
    fire_ncar_intersection.set_geometry(col='geometry')  
    fire_ncar_intersection = fire_ncar_intersection.set_index([str(day_start_hour)+'Z Start Day', 'row', 'col'])
    fire_ncar_intersection=fire_ncar_intersection[~fire_ncar_intersection.index.duplicated()]

    fire_ncar_intersection_xr = fire_ncar_intersection.to_xarray()
    fire_ncar_intersection_xr['weights_mask'] = xr.where(fire_ncar_intersection_xr['weights']>0,1, np.nan)
    print(fire_ncar_intersection_xr)
    #print(fire_ncar_intersection)
    #load in rave data associated with the fire
    times = pd.date_range(np.datetime64(df[str(day_start_hour)+ 'Z Start Day'].iloc[0]),
                        np.datetime64(df[str(day_start_hour)+ 'Z Start Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))
    ncar_filenames,times_back_used = make_file_namelist(times,'/data2/lthapa/YYYY/FMC/fmc_YYYYMMDD_20Z.nc')

    
    dat_ncar = xr.open_mfdataset(ncar_filenames,concat_dim='Time',combine='nested',compat='override', coords='all')
    dat_ncar = dat_ncar.assign_coords({'Time': times_back_used}) #assign coords so we can select in time
    dat_ncar = dat_ncar.reindex(Time=times,method='nearest') #makes the data daily and fills in any gaps
    dat_ncar = dat_ncar.where(dat_ncar!=0) #masks out the 0s for the ocean
    
    #select the locations and times we want
    dat_ncar_sub = dat_ncar.isel(south_north = fire_ncar_intersection_xr['row'].values.astype(int), 
                                 west_east = fire_ncar_intersection_xr['col'].values.astype(int)).sel(
                                 Time = pd.to_datetime(fire_ncar_intersection_xr[str(day_start_hour)+'Z Start Day'].values))#these should be lined up correctly
    print(dat_ncar_sub['FMCG2D'].values)
    
    df_ncar_weighted['day'].iloc[:] = pd.to_datetime(fire_ncar_intersection_xr[str(day_start_hour)+'Z Start Day'].values)
    df_ncar_unweighted['day'].iloc[:] = pd.to_datetime(fire_ncar_intersection_xr[str(day_start_hour)+'Z Start Day'].values)

    for var in varis_ncar[1:]:
        df_ncar_weighted[var] = np.nansum(fire_ncar_intersection_xr['weights'].values*dat_ncar_sub[var].values, axis=(1,2))
        df_ncar_unweighted[var] = np.nansum(fire_ncar_intersection_xr['weights'].values*dat_ncar_sub[var].values, axis=(1,2))

           
    #this day is messed up, fill it in with NANS
    df_ncar_weighted.iloc[df_ncar_weighted['day']=='2020-09-09'] = [pd.date_range(np.datetime64('2020-09-09'),np.datetime64('2020-09-09')+np.timedelta64(0,'D')),
                                                           np.nan,np.nan]
    df_ncar_unweighted.iloc[df_ncar_unweighted['day']=='2020-09-09'] = [pd.date_range(np.datetime64('2020-09-09'),np.datetime64('2020-09-09')+np.timedelta64(0,'D')),
                                                           np.nan,np.nan]
    return df_ncar_weighted, df_ncar_unweighted


path_poly = '/data2/lthapa/ML_daily/fire_polygons/'
suffix_poly = 'Z_Day_Start.geojson'
start_time=12

years =[2019,2020,2021]
#years = [2021]



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
        
        if years[jj]==2019:
            df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-08-01'))&
                              (days<=np.datetime64(str(years[jj])+'-12-30'))].reset_index(drop=True)
        elif years[jj]==2020:
            df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                             (days<=np.datetime64(str(years[jj]+1)+'-01-01'))].reset_index(drop=True)
        elif years[jj]==2021:
            df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                              (days<=np.datetime64(str(years[jj])+'-10-27'))].reset_index(drop=True)
        #print(df_fire)
        
        if len(df_fire)>0:
            ncar_weighted, ncar_unweighted = ncar_timeseries(df_fire, start_time)
            ncar_weighted = pd.concat([ncar_weighted, pd.DataFrame({'irwinID':[fireID]*len(ncar_weighted)})], axis=1)
            ncar_unweighted = pd.concat([ncar_unweighted, pd.DataFrame({'irwinID':[fireID]*len(ncar_unweighted)})], axis=1)
            print(ncar_weighted)
            #print(ncar_unweighted)
        
            ncar_weighted.to_csv('./fire_features_ncar/'+fireID+str(years[jj])+'_Daily_ESI_Weighted_'+str(start_time)+'Z_day_start.csv') #daily averages
            ncar_unweighted.to_csv('./fire_features_ncar/'+fireID+str(years[jj])+'_Daily_ESI_Unweighted_'+str(start_time)+'Z_day_start.csv') #daily averages
        
    

