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

# Import dask
import dask 
import dask.array as da
from dask.distributed import Client

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

# Select the relevant bands depending on the index
def relevantBands(indices):
    if indices == 'NDVI':
        ISMeasurements = ['red', 'nir']
    if indices == 'NDWI_Green':
        ISMeasurements = ['green', 'red']
    if indices == 'NDWI_SWIR':
        ISMeasurements = ['green', 'red','swir1']
    if indices == 'NDDI':
        ISMeasurements = ['red', 'green', 'swir1']
    if indices == 'EVI':
        ISMeasurements = ['red', 'blue']
    return(ISMeasurements)


# Correct NDVI bands
def NDVI(dataset):
    NDVI = (dataset.nir - dataset.red)/(dataset.nir + dataset.red)
    NDVI = NDVI.where(dataset.nir.notnull() & dataset.red.notnull())
    return NDVI

def EVI(dataset):
    #ds = dataset / 10000 # Bands need to be scaled between 0 and 1
    C1 = 6
    C2 = 7.5
    L = 1
    return 2.5 * ((dataset.nir - dataset.red) / (dataset.nir + (C1 * dataset.red) - (C2 * dataset.blue) + L))

def NDWI_Green(dataset):
    return (dataset.green - dataset.nir)/(dataset.green + dataset.nir)

def NDWI_SWIR(dataset):
    return (dataset.green - dataset.swir1)/(dataset.green + dataset.swir1)

def NDDI(dataset):
    aNDVI = NDVI(dataset)
    aNDWI = NDWI_SWIR(dataset)
    return (aNDVI - aNDWI)/(aNDVI + aNDWI)

def run_product(data_dir, product, measurements, dask_chunks, clip_coords, mosaic_type, output_dir, indices):
    
    # Prepare the dataset with all the scenes in the data directory.
    ds = prep_dataset(data_dir, measurements, product, dask_chunks, clip_coords)

    # Use the landsat cloud mask function to mask out clouds.
    clean_mask = ls_clean_mask(ds, keep_water=False)

    # Create composites (summarizing through time) of the baseline and analysis datasets
    ds_composite = create_temporal_composite(ds, mosaic_type, clean_mask = clean_mask)

    # Select and apply desired index
    indices_function = {"NDVI": NDVI, "NDWI_Green": NDWI_Green, "NDWI_SWIR": NDWI_SWIR, "EVI": EVI, "NDDI": NDDI}
    indices_compositor = indices_function[indices]
    indices_composite = indices_compositor(ds_composite)

    # Export the data to desired output directory. 
    indices_composite.rio.to_raster(f"{output_dir}/index_composite.tif", dtype="float32", driver='COG')




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

    mosaic_type = 'median'

    # Set the desired index (options: 'EVI', 'NDVI', 'NDDI', NDWI_Green', 'NDWI_SWIR')
    indices = 'NDVI'

    # Based on desired indices, select relevant bands
    measurements = relevantBands(indices)  

    # Add either the pixel_qa or the scene_classification band to the list of measurements
    if product  in ["sentinel_2"]:
        measurements = measurements + ["nir08", "scene_classification"]
    elif product.startswith('landsat_'):    
        measurements = measurements + ["nir", "pixel_qa"]
    else:
        print("invalid product")

    data_dir = '/home/spatialdays/Documents/ARD_Data/StMaarten_Landsat/'
    output_dir = '/home/spatialdays/Documents/product-notebooks/output/'

    # Running on data from St Maarten
    run_product(data_dir, product, measurements, dask_chunks, clip_coords, mosaic_type, output_dir, indices)