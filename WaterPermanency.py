# Standard imports
import xarray as xr
from datetime import datetime
import yaml
#import rioxarray as rxr
import glob

# Import functions to load and stack data without datacube.
from notebook_functions import *

# Import dask
import dask 
import dask.array as da
from dask.distributed import Client


def run_product(data_dir, product, allmeasurements, dask_chunks, clip_coords, mosaic_type, output_dir):
    
    # Prepare the dataset with all the scenes in the data directory.
    ds = prep_dataset(data_dir, allmeasurements, product, dask_chunks, clip_coords)

    # Generate the water mask (keeping the water).
    water_mask = ls_clean_mask(ds, keep_water=True)

    # Create water composite using the desired mosaic_type
    if mosaic_type in ['mean']:
        water_mosaic = water_mask.mean(dim=['time'])
    elif mosaic_type in ['median']:
        water_mosaic = water_mask.median(dim=['time'])
    else:
        print('invalid mosaic')

    # Export the data to desired output directory. 
    water_mosaic.rio.to_raster(f'{output_dir}/water_permanency.tif', dtype='float32', driver='COG')



if __name__ == "__main__":
    client = Client(n_workers=2, threads_per_worker=4, memory_limit='7GB')

    # Running locally on landsat 8 data for now
    product = 'landsat_8'
    allmeasurements = ["green","red","blue","nir","swir1","swir2", "pixel_qa"]

    # St Maarten bounding box to subset the data
    clip_coords = {'min_lon':-63.461424,
                'min_lat': 17.950000,
                'max_lon': -62.80000,
                'max_lat': 18.334848}

    # Set size of dask chunks to use for the scenes
    dask_chunks = dict(
        x = 1000,
        y = 1000
    )

    mosaic_type = 'mean'

    data_dir = '/home/spatialdays/Documents/ARD_Data/StMaarten_Landsat/'
    output_dir = '/home/spatialdays/Documents/product-notebooks/output/'

    # Running on data from St Maarten
    run_product(data_dir, product, allmeasurements, dask_chunks, clip_coords, mosaic_type, output_dir)