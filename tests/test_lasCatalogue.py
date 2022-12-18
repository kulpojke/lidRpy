#%%
import sys
import os
from time import time

from numba.core.errors import NumbaDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

import matplotlib.pyplot as plt

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

bbox = [6107600.0, 6111499.99, 1832700.0, 1837299.99]
t0 = time()

data_dir = '../test_data'

test_import()

ctg = test_init(data_dir)

test_make_parquets(ctg)

t = time() - t0
print(f'It took {round(t/60, 2)} minutes to build parquets. ')

#df, df2 = test_parquet_reading(ctg)

#t = time() - t
#print(f'It took {round(t/60, 2)} minutes to read all parquets and then read from bbox . ')

ctg.make_chm(bbox,
    save_dir='/home/michael/work/lidRpy/tmp',
    make_dem=True,
    dem_dir='/home/michael/work/lidRpy/dem_tmp'
    )

t = time() - t
print(f'It took {round(t/60, 2)} minutes to build chm and dem from bbox . ')


ctg.tree_detection()

t = time() - t
print(f'It took {round(t/60, 2)} minutes to run tree detection on the chm. ')



# %%

data_dir = '../test_data'

test_import()

ctg = test_init(data_dir)

test_make_parquets(ctg)

t = time() - t0
print(f'It took {round(t/60, 2)} minutes to build parquets. ')


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