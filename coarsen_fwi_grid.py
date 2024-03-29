#import statements
import rasterio
from rasterio.windows import get_data_window,Window, from_bounds
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.enums import Resampling
import pandas as pd
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import datetime
from joblib import Parallel, delayed
import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from matplotlib import path 


#define helper functions
def build_one_coarse_cell(row_select,col_select,ii,jj,loads):
    #print(ii,jj)
    #tic = datetime.datetime.now()
    win = Window(col_select[jj],row_select[ii],33,33)
    #win = Window(col_select,row_select,10,10)
    with rasterio.open('../FCCS_Fuel_Fire_Danger_Metric.tif') as src: 
        w = src.read(1, window=win)
        w_flat = w.flatten()
        row_win = np.arange(win.row_off,win.row_off+win.height)
        col_win = np.arange(win.col_off,win.col_off+win.width)
        COLS_WIN,ROWS_WIN=np.meshgrid(col_win,row_win)
        
        xs_win, ys_win = rasterio.transform.xy(src.transform, ROWS_WIN, COLS_WIN)
    #centers are the middle of the 300mx300m box
    y_ctr =(np.amin(ys_win)+np.amax(ys_win))/2
    x_ctr =(np.amin(xs_win)+np.amax(xs_win))/2
    #print(w_flat)
    #print(loadings.loc[w_flat])
    
    loadings_sub = loads.loc[w_flat].drop(columns=['Unnamed: 0', 'COUNT','FUELBED_NA'])
    df_cell = pd.DataFrame(loadings_sub.mean(axis=0)).transpose()
    df_cell['y_ctr']=y_ctr
    df_cell['x_ctr'] = x_ctr
    df_cell['row'] = ii
    df_cell['col'] = jj

    #toc = datetime.datetime.now()
    #print(toc-tic)
    return(df_cell.set_index(['row','col']))

#LAT and LON are 2d arrays
def calculate_grid_cell_corners(LAT, LON):
    #we will assume the very edges of the polygons don't touch the boundary of the domain
    lat_corners = (LAT[0:(LAT.shape[0]-1),  0:(LAT.shape[1])-1] + LAT[1:(LAT.shape[0]), 1:(LAT.shape[1])])/2
    lon_corners = (LON[0:(LAT.shape[0]-1),  0:(LAT.shape[1])-1] + LON[1:(LAT.shape[0]), 1:(LAT.shape[1])])/2
    return lat_corners, lon_corners

#makes and saves a geodataframe of a grid given the center and corner points for that grid as 2D matrices
def build_grid_netcdf(LAT_COR, LON_COR, LAT_CTR, LON_CTR, filename):
    #loop over the centers
    nrows_center = LAT_CTR.shape[0]
    ncols_center = LAT_CTR.shape[1]
    print(nrows_center, ncols_center)

    nrows_corner = LAT_COR.shape[0]
    ncols_corner = LAT_COR.shape[1]
    print(nrows_corner,ncols_corner)
    
    rows_ctr = np.arange(nrows_center)
    cols_ctr = np.arange(ncols_center)
    rows_cor = np.arange(nrows_corner)
    cols_cor = np.arange(ncols_corner)

    
    dat_grid = xr.Dataset(
        data_vars = dict(
            LAT_CTR=(['rows_ctr','cols_ctr'],LAT_CTR),
            LON_CTR=(['rows_ctr','cols_ctr'],LON_CTR),
            LAT_COR=(['rows_cor','cols_cor'],LAT_COR),
            LON_COR=(['rows_cor','cols_cor'],LON_COR),
        ),
        
        coords = dict(
            rows_ctr =(['rows_ctr'],rows_ctr),
            cols_ctr =(['cols_ctr'],cols_ctr),
            rows_cor =(['rows_cor'],rows_cor),
            cols_cor =(['cols_cor'],cols_cor),
        
        )
        
    )
    print(dat_grid)
    dat_grid.to_netcdf(filename+'.nc')


#fnction is from here: https://gis.stackexchange.com/questions/71630/subsetting-a-curvilinear-netcdf-file-roms-model-output-using-a-lon-lat-boundin
def bbox2ij(lon,lat,bbox):
    bbox=np.array(bbox)
    mypath=np.array([bbox[[0,1,1,0]],bbox[[2,2,3,3]]]).T
    p = path.Path(mypath)
    points = np.vstack((lon.flatten(),lat.flatten())).T
    n,m = np.shape(lon)
    inside = p.contains_points(points).reshape((n,m))
    ii,jj = np.meshgrid(range(m),range(n)) #ii is the columns, jj is the rows
    return min(ii[inside]),max(ii[inside]),min(jj[inside]),max(jj[inside])

loadings = pd.read_csv('FWI_Category_Fuel_Loadings.csv').set_index('VALUE')

#get the dimensions of the whole tif file
src = rasterio.open('../FCCS_Fuel_Fire_Danger_Metric.tif')

print(src.count) #number of bands/data layers
file_width = src.width
file_height = src.height
print(file_width, file_height) #1.5e10 points total!

row_coarse = np.arange(0, file_height,33)
col_coarse = np.arange(0,file_width,33)
print(len(row_coarse), len(col_coarse)) #1.5e8 points

COLS_WIN,ROWS_WIN=np.meshgrid(col_coarse,row_coarse)
xs_win, ys_win = rasterio.transform.xy(src.transform, ROWS_WIN, COLS_WIN)

#transform the coarsened grid into lat/lon, so we can select the bounding box
xs_win, ys_win = rasterio.transform.xy(src.transform, ROWS_WIN, COLS_WIN)

source_crs = 'epsg:5070' # Coordinate system of the file, conus Albers
target_crs = 'epsg:4326' # Global lat-lon coordinate system

conusAlbers_to_latlon = pyproj.Transformer.from_crs(source_crs, target_crs)
tic = datetime.datetime.now()
lat, lon = conusAlbers_to_latlon.transform(np.array(xs_win), np.array(ys_win))
toc = datetime.datetime.now()
print(toc-tic)

#select the bounding box
i0, i1, j0, j1 = bbox2ij(lon, lat, [-124, -101, 31, 49]) #original one
#i0, i1, j0, j1 = bbox2ij(lon, lat, [-124, -103, 31, 49]) #smaller, not as far east

#i0, i1, j0, j1 = bbox2ij(lon, lat, [-118.2, -117.4, 33.7, 34.7]) #bobcat
print(i0, i1, j0, j1)
#print(lon[j0:j1, i0:i1], lat[j0:j1, i0:i1])
row_sel = row_coarse[j0:j1]
col_sel = col_coarse[i0:i1]

print(len(row_sel), len(col_sel))

#make the coarse cells
tic = datetime.datetime.now()
print(tic)
coarse_cells = Parallel(n_jobs=24)(delayed(build_one_coarse_cell)
                                 (row_sel,col_sel,xx,yy,loadings) 
                                 for xx in range(len(row_sel)) for yy in range(len(col_sel)))
toc =datetime.datetime.now()
print(toc-tic)
coarse_cells_df = pd.concat(coarse_cells)

tic = datetime.datetime.now()
coarse_cells_df=coarse_cells_df.reset_index().set_index(['row','col'])
print(coarse_cells_df.iloc[0:1])
coarse_cells_xr = coarse_cells_df.to_xarray()

toc = datetime.datetime.now()
print(toc-tic)
print(coarse_cells_xr)


source_crs = 'epsg:5070' # Coordinate system of the file, conus Albers
target_crs = 'epsg:4326' # Global lat-lon coordinate system

conusAlbers_to_latlon = pyproj.Transformer.from_crs(source_crs, target_crs)
lat, lon = conusAlbers_to_latlon.transform(coarse_cells_xr['x_ctr'].values, coarse_cells_xr['y_ctr'].values)

print(lat.shape)

lat_corners, lon_corners = calculate_grid_cell_corners(lat, lon)
print(lat_corners.shape)


#MAKE SURE THE PLOT LOOKS GOOD BEFORE DOING THIS
#put the grid into a netcdf
build_grid_netcdf(lat_corners, lon_corners, lat, lon, 'FUEL_FWI_GRID_360M')

#save the data in another netcdf
coarse_cells_xr=coarse_cells_xr.assign_coords(lat_ctr=(('row', 'col'), lat), 
                                              lon_ctr=(('row', 'col'), lon))


print(coarse_cells_xr['lon_ctr'].values)

print(coarse_cells_xr)
#coarse_cells_xr.to_netcdf('bobcat_fuel_fwi.nc')
coarse_cells_xr.to_netcdf('fuel_fwi_360m.nc')
