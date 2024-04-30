# Standard imports
import xarray as xr 
from shapely import wkt
from datetime import datetime
import numpy as np
import yaml
import rioxarray as rxr
import glob

# Import functions to load and stack data without datacube.
from notebook_functions import *
from utilities import *

# Import dask
import dask 
import dask.array as da
from dask.distributed import Client

# Functions to generate geomedians
from odc.algo import xr_geomedian, to_rgba, to_f32

def create_temporal_composite(dataset, mosaic_type, clean_mask=None):
    if clean_mask is not None:
        dataset = dataset.where(clean_mask == 1, np.nan)
    
    # Select and run the mosaic type
    if mosaic_type in ['mean']:
        composite = dataset.mean(dim=['time'])
    elif mosaic_type in ['max']:
        composite = dataset.max(dim=['time'])
    elif mosaic_type in ['min']:
        composite = dataset.min(dim=['time'])
    elif mosaic_type in ['median']:
        composite = dataset.median(dim=['time'])
    else:
        print('invalid mosaic')
    
    return composite

def normalize_0_1(band):
    """Normalize band ranges between 0 and 1 for image display purposes."""
    band_min, band_max = (band.min(), band.max())
    return ((band-band_min)/((band_max - band_min)))

def gammacorr(band):
    """Apply gamma correction to the band (brightens image)."""
    gamma=2
    return np.power(band, 1/gamma)


def run_product(data_dir, product, measurements, dask_chunks, clip_coords, output_dir):
    
    # Prepare the dataset with all the scenes in the data directory.
    ds = prep_dataset(data_dir, measurements, product, dask_chunks, clip_coords)

    # Use the landsat cloud mask function to mask out clouds.
    clean_mask = ls_cloud_mask(ds)

    # Create composites (summarizing through time) of the baseline and analysis datasets
    #ds_composite = create_temporal_composite(ds, mosaic_type, clean_mask = clean_mask)

    # Drop the pixel_qa band from the dataset
    ds_analysis = ds.drop(['pixel_qa', 'swir1', 'swir2'])

    ds_clean = ds_analysis.where(clean_mask == 1, np.nan)

    scale, offset = (1, 0)

    ds_clean_32 = to_f32(ds_clean, scale=scale, offset=offset)

    yy = xr_geomedian(ds_clean_32, 
                    num_threads=1,  # disable internal threading, dask will run several concurrently
                    #axis='time',
                    eps=0.2*(1/10_000),  # 1/5 pixel value resolution
                    nocheck=True) 
    
    # Export the data to desired output directory. 
    yy.rio.to_raster(f"{output_dir}/geomedian.tif", dtype="float32", driver='COG')




if __name__ == "__main__":
    client = Client(n_workers=2, threads_per_worker=4, memory_limit='7GB')

    # Running locally on landsat 8 data for now
    product = 'landsat_8'

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

    # Based on desired indices, select relevant bands
    measurements = ["blue", "green", "red", "nir", "swir1", "swir2", "pixel_qa"]  

    data_dir = '/home/spatialdays/Documents/ARD_Data/StMaarten_Landsat/'
    output_dir = '/home/spatialdays/Documents/product-notebooks/output/'

    # Running on data from St Maarten
    run_product(data_dir, product, measurements, dask_chunks, clip_coords, output_dir)