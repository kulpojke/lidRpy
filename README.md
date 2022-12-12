# pyDr
Lidar Forestry and ecology tools

Make a laCatalogue object from directory of las files:
```
ctg = pydr.lasCatalogue(data_dir)
```

At this point `ctg` is just a list of futures.  In order to build the lasCatalogue data we need to read the las files and make parquets.  When reading the las files Height abOve Ground (HAG) is computed.  Points are stored in sorted geo-indexed parquets (Note: At this time they are not goeparquets).

```
ctg.make_parquets()
```

Once the parquets are created an AOI can be read into memory  from a bounding box.

```
bbox = [6107600.0, 6111499.99, 1832700.0, 1837299.99]
aoi = ctg.read_from_bbox(bbox)
```

