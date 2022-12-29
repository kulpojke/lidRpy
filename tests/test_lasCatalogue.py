#%%
import sys
import os
import shutil
from time import time
import scipy.ndimage as ndimage
from skimage.feature import peak_local_max
from osgeo import gdal
from shapely.geometry import Point
import numpy as np
import pandas as pd

from numba.core.errors import NumbaDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import geopandas as gpd

sys.path.append('../src')


def test_import():
    global lidrpy
    import lidrpy


def test_init(data_dir):
    print('Initiating lidrpy instance...')
    ctg = lidrpy.lasCatalogue(data_dir)
    assert len(ctg.files()) == 4
    print('...done.\n')
    return ctg


def test_make_parquets(ctg):
    print('Making parquets...')
    ctg.make_parquets()
    assert len(ctg.files()) == len(os.listdir(ctg.parquet_dir))
    print('...done.\n')


def test_parquet_reading(ctg):
    files = [os.path.join('../parquets/', f) for f in os.listdir('../parquets/')]
    print('Reading parquets...')
    df = ctg.read_parquet(files)
    bbox = [6107600.0, 6111499.99, 1832700.0, 1837299.99]
    df2 = ctg.read_from_bbox(bbox)
    assert len(df) > len(df2) > 0
    print('...done.\n')

    return df, df2



#%%

# clear out old test data
shutil.rmtree('/home/michael/work/lidRpy/test_outputs', ignore_errors=True)

bbox = [462502.0, 463199.4, 3645315.2, 3646392.8]
t = time()

data_dir = '../test_data'

test_import()

ctg = test_init(data_dir)

test_make_parquets(ctg)

t = time() - t
print(f'It took {round(t/60, 2)} minutes to build parquets. ')

#%% ------------------------------------------------------------------------------


ctg.make_chm(bbox,
    save_dir='/home/michael/work/lidRpy/test_outputs/chm_tmp',
    make_dem=True,
    dem_dir='/home/michael/work/lidRpy/test_outputs/dem_tmp'
    )

t = time() - t
print(f'It took {round(t/60, 2)} minutes to build chm and dem from bbox . ')

ctg.filter_chm(
    5,
    save_dir='/home/michael/work/lidRpy/test_outputs/smooth_chm'
    )

t = time() - t
print(f'It took {round(t/60, 2)} minutes to filter with dask . ')

#%% --------------------------------------------------------------------------------
t = time()
save_dir='/home/michael/work/lidRpy/test_outputs/smooth_chm'

for ws in range(1, 20, 2):

    # perform tree detection
    ctg.tree_detection(ws=ws)

    # write a GPKG of ttops
    df = ctg.trees[0]
    df.columns = ['geometry', 'top']
    df = gpd.GeoDataFrame(df)
    df.geometry.to_file(
        f'/home/michael/work/lidRpy/test_outputs/trees_{ws}.gpkg',
        driver='GPKG'
        )

    # perform tree detection with skimage
    df = tree_detection_skimage(ws)


    df = gpd.GeoDataFrame(df)
    df.geometry.to_file(
        f'/home/michael/work/lidRpy/test_outputs/trees_skimage_{ws}.gpkg',
        driver='GPKG'
        )

    t = time() - t
    print(f'It took {round(t, 2)} seconds to run tree detection with ws={ws}. ')
    t= time()







#%%
def tree_detection_skimage(ws):
    raster = ctg.smooth_chm_path
    hmin=3
    resolution = 0.5

    chm_gdal = gdal.Open(raster, gdal.GA_ReadOnly)  
    raster = chm_gdal.GetRasterBand(1).ReadAsArray()

    # get geotransform
    geotransform = chm_gdal.GetGeoTransform()
    x_transform = geotransform[0]
    y_transform = geotransform[3]
    chm_gdal = None

    # use skimage to find the coordinates of local maxima
    tree_maxima = peak_local_max(raster, min_distance=ws, indices=False)

    # remove tree tops lower than minimum height
    tree_maxima[raster <= hmin] = 0

    # label each tree
    tree_markers, num_objects = ndimage.label(tree_maxima)

    # if canopy height is the same for multiple pixels,
    # place the tree top in the center of mass of the pixel bounds
    yx = np.array(
            ndimage.center_of_mass(
                raster, tree_markers, range(1, num_objects+1)
            ), dtype=np.float32
        ) + 0.5
    xy = np.array((yx[:, 1], yx[:, 0])).T

    trees = [Point(*ctg._pixel_to_geo(xy[tidx, 0], xy[tidx, 1], resolution))
            for tidx in range(len(xy))]

    df = pd.DataFrame(np.array([trees, trees], dtype='object').T,
                        dtype='object', columns=['top_cor', 'geometry'])

    df = gpd.GeoDataFrame(df)

    return df

#%%
#%%
#%%
#%%
#%%
#%%

#%%

data_dir = '../test_data'

test_import()

ctg = test_init(data_dir)

ctg.chm_path = '/home/michael/work/lidRpy/test_outputs/chm_tmp/chm.vrt'

t2 = time()

ctg.filter_chm(
    5,
    save_dir='/home/michael/work/lidRpy/test_outputs/smooth_chm/smooth_chm.vrt'
    )



#%%
#%%
#%%
#%%
# %#%%
# %%



#%%
print()
print()
print()
print()
print('----------- tile size tests ---------')
print()
print()
print()
sizes = []
times = []
bbox = [6107600.0, 6111499.99, 1832700.0, 1837299.99]


for ts in range(500, 2500, 500):
    ctg.tile_size = ts

    t = time()
    vrt = ctg.make_chm(bbox)

    t = time() - t
    print(f'It took {round(t/60, 2)} minutes to make chm with {ts} m tiles. ')

    sizes.append(ts)
    times.append(t)


plt.plot(sizes, times);

#%%

# TEST JUST tree_detection WITHOUT ALL THAT SLOW STUFF
data_dir = '../test_data'
vrt = '/home/michael/work/lidRpy/tmp/chm.vrt'


test_import()

ctg = test_init(data_dir)
ctg.chm_path = vrt
ctg.tree_detection()

# HEY! neither chm nor trees have crs