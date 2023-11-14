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


df = #geopandas dataframe of the final August Complex polygon, RANDERSON DATA GOES HERE
    
#Calculate the grid-polygon intersection, in parallel, returns a dataframe
rave_intersections = Parallel(n_jobs=8)(delayed(calculate_intersection)
    (df.iloc[ii:ii+1],'RAVE_GRID_3KM',3000) 
    for ii in range(len(df)))

#PRINT AND SAVE rave_intersections, so you don't have to redo it

print([rave_intersections[jj]['weights'].sum() for jj in range(len(rave_intersections))]) #if all values are 1 (or very close), all intersecting grid cells have been identfied

#puts all the intersections into one geopandas dataframe
fire_rave_intersection=gpd.GeoDataFrame(pd.concat(rave_intersections, ignore_index=True))
fire_rave_intersection.set_geometry(col='geometry')    

#setting the indices for translation to xarray
fire_rave_intersection = fire_rave_intersection.set_index(['row', 'col'])
fire_rave_intersection=fire_rave_intersection[~fire_rave_intersection.index.duplicated()]

fire_rave_intersection_xr = fire_rave_intersection.to_xarray() #convert geopandas to xarray
fire_rave_intersection_xr['weights_mask'] = xr.where(fire_rave_intersection_xr['weights']>0,1, np.nan) #mask of weights to 0s and 1s

#AT THIS POINT, YOU HAVE A MASK, BELOW IS ADVICE FOR HOW TO READ IN RAVE DATA AND APPLY THE MASK

#load in rave data associated with the fire, edit the date range to be for August Complex
times = pd.date_range(np.datetime64(df[str(day_start_hour)+ 'Z Start Day'].iloc[0]),
                        np.datetime64(df[str(day_start_hour)+ 'Z Start Day'].iloc[len(df)-1])+
                        np.timedelta64(1,'D'))                                           
    
#print(rave_filenames)
dat_rave = xr.open_mfdataset(rave_filenames,concat_dim='time',combine='nested',compat='override', coords='all')
dat_rave = dat_rave.resample(time=str(sum_interval)+'H',base=day_start_hour).sum(dim='time') #take the daily sum
    
#select the locations and times we want
dat_rave_sub = dat_rave.isel(grid_yt = fire_rave_intersection_xr['row'].values.astype(int), 
                    grid_xt = fire_rave_intersection_xr['col'].values.astype(int)).sel(
                    time = pd.to_datetime(fire_rave_intersection_xr[str(day_start_hour)+ 'Z Start Day'].values+
                                         'T12:00:00.000000000'))#these should be lined up correctly
ndays = len(fire_rave_intersection_xr[str(day_start_hour)+ 'Z Start Day'])


FRE = np.nansum(fire_rave_intersection_xr['weights'].values*dat_rave_sub['FRE'].values, axis=(1,2))


