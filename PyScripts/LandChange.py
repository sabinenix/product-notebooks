# This one isn't finished yet ... fractional analysis runs but not anomaly calculation




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
from odc.ui import to_png_data


def run_product(baseline_dir, analysis_dir, product, allmeasurements, dask_chunks, clip_coords, output_dir):
    
    # Prepare the dataset with all the scenes in the data directory.
    baseline_ds = prep_dataset(baseline_dir, allmeasurements, product, dask_chunks, clip_coords)
    analysis_ds = prep_dataset(analysis_dir, allmeasurements, product, dask_chunks, clip_coords)

    # Generate the water mask (keeping the water).
    clean_mask_baseline = ls_clean_mask(baseline_ds, keep_water=False)
    clean_mask_analysis = ls_clean_mask(analysis_ds, keep_water=False)


    ds_baseline = baseline_ds.drop(['pixel_qa'])
    ds_analysis = analysis_ds.drop(['pixel_qa'])

    scale, offset = (1,0)

    ds_baseline_clean = ds_baseline.where(clean_mask_baseline == 1, np.nan)
    ds_clean_32_baseline = to_f32(ds_baseline_clean, scale=scale, offset=offset)
    geomedian_baseline = xr_geomedian(ds_clean_32_baseline, 
                    num_threads=1,  # disable internal threading, dask will run several concurrently
                    #axis='time',
                    eps=0.2*(1/10_000),  # 1/5 pixel value resolution
                    nocheck=True) 
    
    ds_analysis_clean = ds_analysis.where(clean_mask_analysis == 1, np.nan)
    ds_clean_32_analysis = to_f32(ds_analysis_clean, scale=scale, offset=offset)
    geomedian_analysis = xr_geomedian(ds_clean_32_analysis, 
                    num_threads=1,  # disable internal threading, dask will run several concurrently
                    #axis='time',
                    eps=0.2*(1/10_000),  # 1/5 pixel value resolution
                    nocheck=True)
    

    # Run fractional cover classifier on the geomedian dataset (baseline)
    geomedian_baseline = geomedian_baseline.rename({'x':'longitude', 'y':'latitude'})
    frac_classes_baseline = frac_coverage_classify(geomedian_baseline)

    # Run fractional cover classifier on the geomedian dataset (analysis)
    geomedian_analysis = geomedian_analysis.rename({'x':'longitude', 'y':'latitude'})
    frac_classes_analysis = frac_coverage_classify(geomedian_analysis)


    # Use stack arrays with composite time handling to align composites before calculations.
    #parameter_analysis_composite, parameter_baseline_composite = stack_arrays([parameter_analysis_composite, parameter_baseline_composite], parameter_analysis_composite, time_handling='Composites')


    # Calculate the difference in fractional cover classes
    parameter_anomaly_bs = frac_classes_analysis.bs - frac_classes_baseline.bs
    parameter_anomaly_pv = frac_classes_analysis.pv - frac_classes_baseline.pv
    parameter_anomaly_npv = frac_classes_analysis.npv - frac_classes_baseline.npv

    # Export the data to desired output directory. 
    parameter_anomaly_bs.rio.to_raster(f"{output_dir}/land_change_bs.tif", dtype="float32", driver='COG')
    parameter_anomaly_pv.rio.to_raster(f"{output_dir}/land_change_pv.tif", dtype="float32", driver='COG')
    parameter_anomaly_npv.rio.to_raster(f"{output_dir}/land_change_npv.tif", dtype="float32", driver='COG')



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

    baseline_dir = '/home/spatialdays/Documents/ARD_Data/StMaarten_Landsat_baseline/'
    analysis_dir = '/home/spatialdays/Documents/ARD_Data/StMaarten_Landsat_analysis/'

    output_dir = '/home/spatialdays/Documents/product-notebooks/output/'

    # Running on data from St Maarten
    run_product(baseline_dir, analysis_dir, product, allmeasurements, dask_chunks, clip_coords, output_dir)