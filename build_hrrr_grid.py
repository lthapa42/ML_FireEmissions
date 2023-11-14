import pandas as pd
#pd.set_option('display.max_rows', None)
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
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import cascaded_union
from datetime import datetime, timedelta
import time
import warnings
import statsmodels.api as sm
import datetime
import math
from scipy.ndimage.interpolation import shift
import shapely.wkt
from shapely.validation import explain_validity
from shapely.validation import make_valid
import xarray as xr

warnings.filterwarnings('ignore')
import seaborn as sns

#makes and saves a geodataframe of a grid given the center and corner points for that grid as 2D matrices
def build_grid_gdf(LAT_COR, LON_COR, LAT_CTR, LON_CTR, filename):
    #loop over the centers
    nrows = LAT_CTR.shape[0]
    ncols = LAT_CTR.shape[1]
    print(nrows, ncols)

    #preallocate the dataframe
    df_size = nrows*ncols
    df_grid = gpd.GeoDataFrame({'lat': np.zeros(df_size), 
                                'lon': np.zeros(df_size),
                                'row':np.zeros(df_size),
                                'col':np.zeros(df_size),
                                'geometry':np.zeros(df_size)})

    count=0
    for ii in range(nrows):
        for jj in range(ncols):
            #print(ii,jj,count)
            #print(LAT_CTR[ii,jj], LON_CTR[ii,jj]) #ctr
            sw = (LON_COR[ii, jj],LAT_COR[ii, jj]) #SW
            se =(LON_COR[ii, jj+1],LAT_COR[ii, jj+1]) #SE
            nw = (LON_COR[ii+1, jj],LAT_COR[ii+1, jj]) #NW
            ne = (LON_COR[ii+1, jj+1],LAT_COR[ii+1, jj+1]) #NE
            
            poly_cell = Polygon([sw,nw,ne,se])
            df_grid.iloc[count,:] = [LAT_CTR[ii,jj], LON_CTR[ii,jj],ii,jj,poly_cell]

            
            count=count+1
    print(df_grid)
    df_grid.set_geometry(col='geometry', inplace=True)
    df_grid.to_file(filename+'.geojson', driver='GeoJSON')



path_hrrr = '/data2/lthapa/ML_daily/pygraf/Processed_HRRR_2020091800.nc'
dat_hrrr = xr.open_dataset(path_hrrr)

lat_centers = dat_hrrr['grid_latt'].values #1057x1797
lon_centers = dat_hrrr['grid_lont'].values #1057x1797

#drop the outer ring of lat/lons
lat_centers = lat_centers[1:lat_centers.shape[0],1:lat_centers.shape[1]] #1056x1796
lon_centers = lon_centers[1:lon_centers.shape[0],1:lon_centers.shape[1]] #1056x1796

#load in the corners
lat_corners = dat_hrrr['grid_lat'].values #1057x1797
lon_corners = dat_hrrr['grid_lon'].values #1057x1797

build_grid_gdf(lat_corners, lon_corners, lat_centers, lon_centers, 'HRRR_GRID')
