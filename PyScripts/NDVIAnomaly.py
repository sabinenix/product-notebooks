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

def NDVI(dataset):
    NDVI = (dataset.nir - dataset.red)/(dataset.nir + dataset.red)
    NDVI = NDVI.where(dataset.nir.notnull() & dataset.red.notnull())
    return NDVI

def run_product(baseline_dir, analysis_dir, product, allmeasurements, dask_chunks, clip_coords, mosaic_type, output_dir, water_thresh=0.3):
    
    # Prepare the dataset with all the scenes in the data directory.
    baseline_ds = prep_dataset(baseline_dir, allmeasurements, product, dask_chunks, clip_coords)
    analysis_ds = prep_dataset(analysis_dir, allmeasurements, product, dask_chunks, clip_coords)

    # Use the landsat cloud mask function to mask out clouds.
    clean_mask_baseline = ls_cloud_mask(baseline_ds)
    clean_mask_analysis = ls_cloud_mask(analysis_ds)

    # Generate the water mask (keeping the water).
    water_mask_baseline = ls_clean_mask(baseline_ds, keep_water=True)
    water_mask_analysis = ls_clean_mask(analysis_ds, keep_water=True)

    # Create composites (summarizing through time) of the baseline and analysis datasets
    baseline_composite = create_temporal_composite(baseline_ds, mosaic_type, clean_mask = clean_mask_baseline)
    analysis_composite = create_temporal_composite(analysis_ds, mosaic_type, clean_mask = clean_mask_analysis)

    # Generate the water composites to use for water masking 
    water_composite_baseline = create_temporal_composite(water_mask_baseline, 'mean')
    water_composite_analysis = create_temporal_composite(water_mask_analysis, 'mean')

    # Use the composites to mask water out of the baseline and analysis composites
    baseline_composite = baseline_composite.where((baseline_composite != np.nan) & (water_composite_baseline <= water_thresh))
    analysis_composite = analysis_composite.where((analysis_composite != np.nan) & (water_composite_analysis <= water_thresh))


    # Calculate NDVI composite for both datasets
    ndvi_baseline_composite = NDVI(baseline_composite)
    ndvi_analysis_composite = NDVI(analysis_composite)

    # Use stack arrays with composite time handling to align composites before calculations.
    ndvi_analysis_composite, ndvi_baseline_composite = stack_arrays([ndvi_analysis_composite, ndvi_baseline_composite], ndvi_analysis_composite, time_handling='Composites')

    # Calculate NDVI Anomaly.
    ndvi_anomaly = ndvi_analysis_composite - ndvi_baseline_composite

    # Export the data to desired output directory. 
    ndvi_analysis_composite.rio.to_raster(f"{output_dir}/ndvi_analysis.tif", dtype="float32", driver='COG')
    ndvi_baseline_composite.rio.to_raster(f"{output_dir}/ndvi_baseline.tif", dtype="float32", driver='COG')
    ndvi_anomaly.rio.to_raster(f"{output_dir}/ndvi_anomaly.tif", dtype="float32", driver='COG')



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
    water_thresh = 0.3

    baseline_dir = '/home/spatialdays/Documents/ARD_Data/StMaarten_Landsat_baseline/'
    analysis_dir = '/home/spatialdays/Documents/ARD_Data/StMaarten_Landsat_analysis/'

    output_dir = '/home/spatialdays/Documents/product-notebooks/output/'

    # Running on data from St Maarten
    run_product(baseline_dir, analysis_dir, product, allmeasurements, dask_chunks, clip_coords, mosaic_type, output_dir, water_thresh=water_thresh)