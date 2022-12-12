#%%
import sys
import os
from time import time

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


t0 = time()

data_dir = '../test_data'
test_import()
ctg = test_init(data_dir)
test_make_parquets(ctg)

t1 = time() - t0
print(f'It took {round(t1/60, 2)} minutes to build parquets. ')

files = [os.path.join('../parquets/', f) for f in os.listdir('../parquets/')]
df = ctg.read_parquet(files)
bbox = [6107600.0, 6111499.99, 1832700.0, 1837299.99]
df2 = ctg.read_from_bbox(bbox)
assert len(df) > len(df2) > 0

df2.HeightAboveGround.hist(bins=50)

# %%
# Next steps:
# Use dalponte from pytcrown, will need to make chm first, maybe just use ppdal for that? Or make a HAG filter.