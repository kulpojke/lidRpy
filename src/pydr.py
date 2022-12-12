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


#%%

class lasCatalogue:
    '''docstring'''

    def __init__(self, data_dir, tilesize=500):
        self.data_dir = data_dir
        self.parquet_dir = os.path.join(
            os.path.dirname(self.data_dir),
            'parquets')
        self.__inbox_func = None
        self.tilesize = tilesize
        self.files = []
        self.__see_what_files(self.data_dir)
        self.lazy = []

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

        # no need for more cpu's then files
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
        print(f'number of files: {len(files)}')
        df = self.read_parquet(files)

        func = self. __make_inbox_func(bbox)

        arr = df[['X', 'Y']].to_numpy()
        mask = func(arr[:,0], arr[:,1])

        df = df[mask]

        return df


def make_chm(self, bbox, tilesize=1000):
    '''
    
    '''
    # break bbox up into buffered tiles

    # makes chm from tiles, put into a directory

    # build a vrt wfrom overlaping tiles

    # return vrt