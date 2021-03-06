# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:25:53 2019

@author: VanBoven
"""

import os
import rasterio.mask
import json
import geopandas as gpd

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def clip_ortho2plot(ortho_path, filename, shp_path, output_path):
    with rasterio.open(os.path.join(ortho_path)) as src:
        #read shapefile
        gdf = gpd.read_file(shp_path)
        coords = getFeatures(gdf)
        out_image, out_transform = rasterio.mask.mask(src, coords, crop=True)
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
    with rasterio.open(os.path.join(output_path, filename), \
    "w", **out_meta, BIGTIFF='YES',NUM_THREADS='ALL_CPUS',COMPRESS='LZW') as dest:
        dest.write(out_image)
