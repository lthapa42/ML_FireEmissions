import xarray as xr
import scipy.io
from scipy.interpolate import interp1d
import numpy as np
import numpy.matlib as npm
from matplotlib import path
import math
import pandas as pd
import os
from datetime import timedelta
from os.path import exists
from shapely.geometry import LineString,Polygon,Point
from shapely.validation import explain_validity

import shapely.wkt

import datetime
import json

import warnings
warnings.filterwarnings('ignore')

#libraries i added to pygraf
from joblib import Parallel, delayed
import geopandas as gpd #pip version, also pip installed rtree
import matplotlib.pyplot as plt



#makes and saves a geodataframe of a grid given the center and corner points for that grid as 2D matrices

def build_one_gridcell(LAT_COR, LON_COR, LAT_CTR, LON_CTR, loc):
    ii=loc[0]
    jj=loc[1]

    #print(LAT_CTR[ii,jj], LON_CTR[ii,jj]) #ctr
    sw = (LON_COR[ii, jj],LAT_COR[ii, jj]) #SW
    se =(LON_COR[ii, jj+1],LAT_COR[ii, jj+1]) #SE
    nw = (LON_COR[ii+1, jj],LAT_COR[ii+1, jj]) #NW
    ne = (LON_COR[ii+1, jj+1],LAT_COR[ii+1, jj+1]) #NE
            
    poly_cell = Polygon([sw,nw,ne,se])


    return LAT_CTR[ii,jj], LON_CTR[ii,jj],ii,jj,poly_cell

    

#poly is the polygon for one timestep (in lambert conformal conic)

#dataset_name is the name of a model grid nc file

#bf is the size of the polygon buffer



def calculate_intersection(poly,dataset_name,bf):
    #load in the merra grid

    grid = xr.open_dataset(dataset_name+'.nc')

    

    #get the bounds of the buffered polygons

    #poly_latlon =poly#.to_crs(epsg=4326)

    #bounds = poly_latlon.buffer(bf).bounds

    bounds = poly.buffer(bf).to_crs(epsg=4326).bounds



    #first check for rows and cols, filtering near the polygon

    [rows,cols] = np.where((grid.LAT_CTR.values>bounds['miny'].values)&

                    (grid.LAT_CTR.values<bounds['maxy'].values)&

                    (grid.LON_CTR.values>bounds['minx'].values)&

                    (grid.LON_CTR.values<bounds['maxx'].values))

    locs = zip(rows,cols)





    #make a geodataframe (in paralell of the rows and cols)

    results = Parallel(n_jobs=8)(delayed(build_one_gridcell)

                                 (grid['LAT_COR'].values, grid['LON_COR'].values,

                                  grid['LAT_CTR'].values, grid['LON_CTR'].values,loc) 

                                 for loc in locs)



    #format the grid subset into a dataframs

    df_grid=gpd.GeoDataFrame(results)

    df_grid.columns = ['lat', 'lon', 'row', 'col', 'geometry']

    df_grid.set_geometry(col='geometry',inplace=True,crs='EPSG:4326') #need to say it's in lat/lon before transform to LCC

    df_grid=df_grid.to_crs(epsg=3347)



    #intersect the polygon with the grid subset

    intersection = gpd.overlay(df_grid, poly, how='intersection',keep_geom_type=False).drop_duplicates()

    intersection['grid intersection area (ha)'] =intersection['geometry'].area/10000

    intersection['weights'] = intersection['grid intersection area (ha)']/intersection['fire area (ha)'] 

    

    return intersection





#LAT and LON are 2d arrays

def calculate_grid_cell_corners(LAT, LON):

    #we will assume the very edges of the polygons don't touch the boundary of the domain

    lat_corners = (LAT[0:(LAT.shape[0]-1),  0:(LAT.shape[1])-1] + LAT[1:(LAT.shape[0]), 1:(LAT.shape[1])])/2

    lon_corners = (LON[0:(LAT.shape[0]-1),  0:(LAT.shape[1])-1] + LON[1:(LAT.shape[0]), 1:(LAT.shape[1])])/2

    return lat_corners, lon_corners


def make_file_namelist(time,base_filename):

    filename_list = np.array([])

    times_back_used = np.array([])

    for jj in range(len(time)):

        fname = base_filename.replace('YYYY',time[jj].strftime('%Y')).\
                                replace('MM',time[jj].strftime('%m')).\
                                replace('DD',time[jj].strftime('%d')).\
                                replace('HH',time[jj].strftime('%H')).\
                                replace('JJJ',time[jj].strftime('%j'))

        #print(fname)

        if exists(fname):

            filename_list = np.append(filename_list,fname)

            times_back_used = np.append(times_back_used,time[jj])

    return filename_list, times_back_used


def generate_df(variables, length):
    df = pd.DataFrame()
    for vv in variables:
        df[vv] = np.zeros(length)
    return df
