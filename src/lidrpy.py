#%%
import pdal
import dask.dataframe as dd
import dask.array as da
import dask_geopandas
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pyarrow as pa
import pyarrow.parquet as pq
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import os
import shutil
from pathlib import Path
import json

from functools import partial
import pyarrow as pa
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
from numba import vectorize, float32, float64, boolean

import tempfile
from osgeo import gdal

import time
import platform
import warnings
from math import floor
from pathlib import Path


import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.spatial.distance import cdist

from skimage.segmentation import watershed
from skimage.filters import threshold_otsu
# from skimage.feature import peak_local_max

from shapely.geometry import mapping, Point, Polygon

from rasterio.features import shapes as rioshapes

import fiona
from fiona.crs import from_epsg

#%%
class NoTreesException(Exception):
    """ Raised when no tree detected """
    pass


class lasCatalogue:
    '''docstring'''

    def __init__(self, data_dir, tile_size=2000):
        self.data_dir = data_dir
        self.parquet_dir = os.path.join(
            os.path.dirname(self.data_dir),
            'parquets')
        self.__inbox_func = None
        self.tile_size = tile_size
        self.buffer = 15
        self.resolution = 0.5
        self.chm_path = None
        self.files = []
        self.__see_what_files(self.data_dir)
        self.lazy = []
        self.trees = []

        # transient to hold the pq file names from __las_to_parquet
        # for use in __make_parquet_map
        self.parquet_list_ = []
        
        for las in self.files():
            self.lazy.append(self.__las_to_parquet(las))


    def __read_las(self, las, hag=True):
        '''
        Reads las/z, returns df, tile. Where:
            - df is a Pandas DataFrame of points with dimensions as columns.
            - tile is extent information of the las as a string of format 
              'xmin_xmax_ymin_ymax_'
            
        parameters:
            - las - path to las/z file to read.
            - hag - if True, calculates new dimension, HeightAboveGround, during read.
        '''
        # make pipe, execute, extract array
        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader.las(filename=las)
        #pipeline |= pdal.Filter.outlier()
        if hag:
            pipeline |= pdal.Filter.hag_nn(
                count=2,
                allow_extrapolation=True
            )
        n = pipeline.execute()
        arr = pipeline.arrays[0]

        # make into df
        df = pd.DataFrame(arr)
        
        # sort values, make tile name, multi-index 
        df = df.sort_values(['X','Y', 'Z'])
        tile = f'{df.X.min()}_{df.X.max()}_{df.Y.min()}_{df.Y.max()}_'

        # TODO: probably should make tile from extent in 
        # las header, not sure how robust this method is

        return df, tile


    def __see_what_files(self, data_dir, dimensions='all'):
        '''Finds supported files in data_dir, returns list as self.files.'''

        # read the directory
        data = os.listdir(data_dir)

        # look for copcs, laz and las
        copc = [os.path.join(data_dir, d) for 
                d in data if 
                d.endswith('.copc.laz')]

        las  = [os.path.join(data_dir, d) for 
                d in data
                if (not d.endswith('.copc.laz') and
                    d.endswith('.laz')) or
                d.endswith('las')]

        # until a copc reader is implemented just make excuse
        if len(copc) > 0:
            print('COPC reader not yet implemented. Reading COPC files as las.')
            las = las + copc

        self.files = las.copy


    @delayed
    def __las_to_parquet(self, las):
        '''Appends future parquet writes to self.lazy'''
        
        df, tile = self.__read_las(las)

        # path to tile
        tile_path = os.path.join(self.parquet_dir, f'{tile}.parquet')
        self.parquet_list_.append(tile_path)

        # append the to parquet
        return df.to_parquet(tile_path)


    def __make_parquet_map(self):
        '''
        Makes df of parquet tile layout.
        df columns:
            - 'file' - contains 'path/to/parquet/xmin_xmax_ymin_ymax_.parquet'

        '''
        df = pd.DataFrame()
        df['file'] = self.parquet_list_
        df['tile'] = [t[-1] for t in df.file.str.split('/')]

        # make array of lists into 2d array
        arr = np.array([l for l in df.tile.str.split('_').values])


        df['xmin']  = np.float64(arr[:, 0])
        df['xmax']  = np.float64(arr[:, 1])
        df['ymin'] = np.float64(arr[:, 2])
        df['ymax'] = np.float64(arr[:, 3])

        df = df.sort_values(['xmin', 'ymax'])
        df = df.set_index(['xmin', 'ymax'])

        self.tile_map = df[['xmax', 'ymin', 'tile', 'file']]


    def __find_needed_parquets(self, box):
        '''
        returns list of only the parquets on which box falls
        '''
        files = []

        for (xmin, ymax), row in self.tile_map.iterrows():
            if (
                (box[0] >= xmin <= box[1]) |
                (box[0] >= row.xmax <= box[1]) |
                (box[2] >= row.ymin <= box[3]) |
                (box[2] >= ymax <= box[3])
            ):

                files.append(row.file)                

        return files


    def __make_inbox_func(self, bbox):
        '''
        returns a function that checks if point is within 
        bbox.  
        '''
        @vectorize([boolean(float64,float64)])
        def inbox(x, y):
            return(
            x >= bbox[0] and
            x <  bbox[1] and
            y >= bbox[2] and
            y <  bbox[3]
            )
        
        return inbox


    def __read_parquet_wrapper(self, files, columns=None):
        '''
        wrapper to pass to a ProcessPoolExecutor to read parquet files.
        Explicitly enables multithreaded column reading.
        Returns Pandas df.
        '''
        df = pd.read_parquet(
                files,
                columns=columns,
                engine="pyarrow",
                use_threads=True
            )

        return df


    def read_parquet(self,
                       files,
                       columns=None,
                       parallel=True,
                       n_concurrent_files=4,
                       n_concurrent_columns=4,
                       show_progress=True,
                       ignore_index=False,
                       chunksize=None):

        '''
        Reads a list of parquet files and
        returns a single Pandas DataFrame.
        '''

        # no need for more CPUs then files
        if len(files) < n_concurrent_files:
            n_concurrent_files = len(files)

        # no need for more workers than columns
        if columns:
            if len(columns) < n_concurrent_columns:
                n_concurrent_columns = len(columns)
                
        # set number of threads used for reading columns of each parquet
        pa.set_cpu_count(n_concurrent_columns)

        # if more files than useable cpus...
        if (chunksize is None) and (len(files) > n_concurrent_files):
            # divide the files up amongst useable cpus
            chunksize, remainder = divmod(len(files), n_concurrent_files)
            if remainder:
                chunksize += 1
        # otherwise just keep each file in 1 chunk
        else:
            chunksize = 1

        if parallel:

            dfs = thread_map(
                self.__read_parquet_wrapper,
                files,
                max_workers=n_concurrent_files,
                chunksize=chunksize
            )

        else:
            dfs = [self.__read_parquet_wrapper(file)
                   for file
                   in tqdm(files,
                           disabled=not show_progress)]

        # reduce the list of dataframes to a single dataframe
        df = pd.concat(dfs, ignore_index=ignore_index)

        return df


    def __buffer_tiles(self, bbox):
        '''Returns a list of  of buffered tiles as [xmin, xmax, ymin, ymax]'''
        # break bbox up into buffered tiles
        x_grid = np.arange(bbox[0], bbox[1], self.tile_size)
        y_grid = np.arange(bbox[2], bbox[3], self.tile_size)

        boxes = []
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                if i == 0:
                    b0 = x
                else:
                    b0 = x - self.buffer
                
                b1 = x + self.tile_size + self.buffer

                if j == 0:
                    b2 = y
                else:
                    b2 = y - self.buffer

                b3 = y + self.tile_size + self.buffer

                boxes.append([b0, b1, b2, b3])

        return boxes


    def __df_to_xyHAG_arr(self, df):
        '''
        Takes a pandas df, returns a Numpy array of only X, Y, HAG:
        
        array([[ x0, y0, hag0 ],
               [ x1, y1, hag1 ],
               ...,
               [ xn, yn, hagn ]])
        
        for for use with open3d or sklearn (or other).
        '''
        #TODO: is this wrapper pointless?
        return df[['X', 'Y', 'HeightAboveGround']].to_numpy()


    
    def __df_to_structured_arr(self, df):
        '''
        Takes a pandas df, returns a Numpy structured array suitable
        for feeding to pdal
        '''
        s = df.dtypes
        arr = np.array(
            [tuple(x) for x in df.values],
            dtype=list(zip(s.index, s)))

        return arr


    @staticmethod
    def _get_kernel(radius=5, circular=False):
        """ returns a block or disc-shaped filter kernel with given radius
        Parameters
        ----------
        radius :    int, optional
                    radius of the filter kernel
        circular :  bool, optional
                    set to True for disc-shaped filter kernel, block otherwise
        Returns
        -------
        ndarray
            filter kernel
        """
        if circular:
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            return x**2 + y**2 <= radius**2
        else:
            return np.ones((int(radius), int(radius)))


    def _pixel_to_geo(self, pix_x, pix_y, resolution):
        ''' Convert pixel coordinates to projected coordinates
        '''
        x = self.x_transform + (pix_x * resolution)
        y = self.y_transform - (pix_y * resolution)
        return x, y
    # -------------------- public methods ------------------------------------

    def make_parquets(self):
        '''Computes the parquet futures'''

        if os.path.isdir(self.parquet_dir):
            shutil.rmtree(self.parquet_dir)

        os.makedirs(self.parquet_dir)

        with ProgressBar():
            _ = compute(*self.lazy)

        self.__make_parquet_map()


    def read_from_bbox(self, bbox):
        ''''''
        files = self.__find_needed_parquets(bbox)
        df = self.read_parquet(files)

        func = self. __make_inbox_func(bbox)

        arr = df[['X', 'Y']].to_numpy()
        mask = func(arr[:,0], arr[:,1])

        df = df[mask]

        return df


    def make_chm(self, bbox, save_dir=None):
        '''
        TODO: should we use the same tile_size for tiffs?
        '''
        # buffer tiles
        buf_tiles = self.__buffer_tiles(bbox)

        # make place to save
        if not save_dir:
            # make a tmp directory that will be destroyed when we are done
            self._chm_tmp = tempfile.TemporaryDirectory()
            save_dir = self._chm_tmp.name 
        os.makedirs(save_dir, exist_ok=True)
        
        for i, tile in enumerate(buf_tiles):
            # read points in tile
            df = self.read_from_bbox(tile)
            # make df into arr for pdal
            arr = self.__df_to_structured_arr(df)
            # make path 
            filename=os.path.join(save_dir, f'chm_{i}.tif')
            # make pipe
            pipeline = pdal.Writer.gdal(
                filename=filename,
                resolution=self.resolution,
                output_type='mean',
                dimension='HeightAboveGround'
                ).pipeline(arr)
            pipeline.execute()

        # build a vrt from overlaping tiles
        vsi_hrefs = [os.path.join(save_dir, f) for f in os.listdir(save_dir)]
        vrt = os.path.join(save_dir, 'chm.vrt')
        _ = gdal.BuildVRT(vrt, vsi_hrefs)
        _ = None

        self.chm_path = vrt
                
        return vrt


    def tree_detection(self, raster=None, resolution=None, ws=5, hmin=3,
                        return_trees=False, ws_in_pixels=False):
            ''' Detect individual trees from CHM raster based on a maximum filter.
            Identified trees are either stores as list in the tree dataframe or
            returned as ndarray.
            Parameters
            ----------
            raster :        str, optional
                            path to raster of height values (e.g., CHM)
            resolution :    int, optional
                            resolution of raster in m
            ws :            float
                            moving window size (in metre) to detect the local maxima
            hmin :          float
                            Minimum height of a tree. Threshold below which a pixel
                            or a point cannot be a local maxima
            return_trees :  bool
                            set to True if detected trees shopuld be returned as
                            ndarray instead of being stored in tree dataframe
            ws_in_pixels :  bool
                            sets ws in pixel
            Returns
            -------
            ndarray (optional)
                nx2 array of tree top pixel coordinates

            Modified fom:
            Zörner, J.; Dymond, J.; Shepherd J.; Jolly, B.
            PyCrown - Fast raster-based individual tree segmentation for LiDAR data.
            Landcare Research NZ Ltd. 2018, https://doi.org/10.7931/M0SR-DN55
            '''

            # read raster as array
            raster = raster if raster else self.chm_path
            chm_gdal = gdal.Open(raster, gdal.GA_ReadOnly)  
            raster = chm_gdal.GetRasterBand(1).ReadAsArray()


            # get geotransform
            geotransform = chm_gdal.GetGeoTransform()
            self.x_transform = geotransform[0]
            self.y_transform = geotransform[3]
            chm_gdal = None

            # TODO: should we just read the resolution from the raster?
            resolution = resolution if resolution else self.resolution

            if not ws_in_pixels:
                if ws % resolution:
                    raise Exception("Image filter size not an integer number. Please check if image resolution matches filter size (in metre or pixel).")
                else:
                    ws = int(ws / resolution)

            # Maximum filter to find local peaks
            raster_maximum = filters.maximum_filter(
                raster, footprint=self._get_kernel(ws, circular=True))
            tree_maxima = raster == raster_maximum

            # remove tree tops lower than minimum height
            tree_maxima[raster <= hmin] = 0

            # label each tree
            self.tree_markers, num_objects = ndimage.label(tree_maxima)

            if num_objects == 0:
                raise NoTreesException

            # if canopy height is the same for multiple pixels,
            # place the tree top in the center of mass of the pixel bounds
            yx = np.array(
                    ndimage.center_of_mass(
                        raster, self.tree_markers, range(1, num_objects+1)
                    ), dtype=np.float32
                ) + 0.5
            xy = np.array((yx[:, 1], yx[:, 0])).T

            trees = [Point(*self._pixel_to_geo(xy[tidx, 0], xy[tidx, 1], resolution))
                    for tidx in range(len(xy))]

            df = pd.DataFrame(np.array([trees, trees], dtype='object').T,
                                dtype='object', columns=['top_cor', 'top'])
            self.trees.append(df)

            if return_trees:
                return np.array(trees, dtype=object), xy
            
            