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

def resources_timeseries(df,day_start_hour, sit209_data):
    #sit209_data = pd.read_csv('../merged_sit.csv')
    
    
    #get the fire incident number, lat, and lon
    incident_number = df['irwinID'].iloc[0]
    fire_lat = df['Lat Fire'].iloc[0]
    fire_lon = df['Lon Fire'].iloc[0]
    #print(incident_number, fire_lat, fire_lon)
    
    sit209_data_fire = sit209_data[sit209_data['IRWIN_IDENTIFIER']==incident_number]
    print(sit209_data_fire)
    #print(sit209_data_fire.columns.values)
    #do the time zone conversion
    obj=TimezoneFinder() #initialize the timezone finder
    tz = obj.timezone_at(lng=fire_lon, lat=fire_lat) #get the timezone
    local = pytz.timezone(tz)
    utc = pytz.utc
    
    #put the start and end times in local time
    loc_dt_start = [local.localize(datetime.datetime.strptime(date, '%m/%d/%Y %H:%M:%S %p')) for date in sit209_data_fire['REPORT_FROM_DATE'].values]
    loc_dt_end = [local.localize(datetime.datetime.strptime(date, '%m/%d/%Y %H:%M:%S %p')) for date in sit209_data_fire['REPORT_TO_DATE'].values]
    
    #put them in UTC time
    utc_dt_start = [time_start.astimezone(utc) for time_start in loc_dt_start]
    utc_dt_end = [time_end.astimezone(utc) for time_end in loc_dt_end]
    
    start_day = pd.to_datetime(utc_dt_start[0]).strftime('%Y-%m-%d')+' '+str(day_start_hour)+':00'
    
    
    #reassign to UTC time, this DOES keep track of daylight savings (eg +7 is used for PDT, +8 is used for PST)
    sit209_data_fire['Report Start UTC'] = pd.to_datetime(utc_dt_start)
    sit209_data_fire['Report End UTC'] = pd.to_datetime(utc_dt_end)
    sit209_data_fire['Timezone']= tz
    
    #localise the index
    sit209_data_fire = sit209_data_fire.set_index(['Report Start UTC']).tz_localize(None)
    #print(sit209_data_fire.iloc[0:4])
    
    
    ## do the 12z-12z day grouping, based on the UTC times
    #start_day_utc = str(utc_dt_start[0])
    start_day_utc=str(df[str(day_start_hour)+'Z Start Day'][0])
    start_datetime_utc = np.datetime64(start_day_utc[0:10]+'T'+str(day_start_hour).zfill(2)+':00')
    print(start_datetime_utc)
    #sit209_data_fire = sit209_data_fire.resample('24H',origin=start_datetime_utc)

    #personnel = sit209_data_fire['RESOURCE_PERSONNEL'].resample('24H',origin=start_datetime_utc).sum().reset_index()
    percent_contained = sit209_data_fire['PCT_CONTAINED_COMPLETED'].resample('24H',origin=start_datetime_utc).mean().reset_index()
    df_sit209 = pd.concat([percent_contained],axis=1)
    df_sit209.columns=['day', 'percent_contained']
    df_sit209['day'] = pd.to_datetime(df_sit209['day'].values).strftime('%Y-%m-%d')
    df_sit209=df_sit209.fillna(method='ffill')
    #df_sit209['day'].iloc[:] = pd.to_datetime(df[str(day_start_hour)+ 'Z Start Day'].values)

    #inds = df_sit209['day'].isin(df[str(day_start_hour)+'Z Start Day']).values
    
    return df_sit209
    

path_poly = '/data2/lthapa/ML_daily/fire_polygons/'
suffix_poly = 'Z_Day_Start.geojson'
start_time=12
resources_file  = '../YYYY_PCT_CONT.xlsx'

#years =[2019,2020]
years = [2020]



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
    
   
    resources_all = pd.read_excel(resources_file.replace('YYYY', str(years[jj]))).iloc[:,0:6]
    resources_all.columns = ['INC209R_IDENTIFIER','REPORT_FROM_DATE','REPORT_TO_DATE','PCT_CONTAINED_COMPLETED',
                             'INCIDENT_NAME', 'IRWIN_IDENTIFIER']
    print(resources_all.head())

    esi_weighted_all = pd.DataFrame() 
    esi_unweighted_all = pd.DataFrame()
    
    for ii in range(len(irwinIDs)):
        print(ii)
        fireID=irwinIDs[ii]
        print(fireID)
        df_fire = fire_daily[fire_daily['irwinID']==fireID] #this is what gets fed to the feature selection code
        
        #maybe we filter 12Z Start dates to where we have data?
        days=np.array(df_fire['12Z Start Day'].values, dtype='datetime64')
        df_fire = df_fire[(days>=np.datetime64(str(years[jj])+'-07-01'))&
                              (days<=np.datetime64(str(years[jj]+1)+'-01-01'))].reset_index(drop=True)
        resources = resources_timeseries(df_fire, start_time, resources_all)
        
        print(resources)
        resources.to_csv('./fire_features_resources/'+fireID+str(years[jj])+'_Daily_RESOURCES_'+str(start_time)+'Z_day_start.csv') #daily averages

