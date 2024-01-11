import xarray as xr 
import numpy as np

def rename_bands(in_xr, des_bands, position):
    """
    (From genprepWater.py)
    """
    in_xr.name = des_bands[position]
    return in_xr

def stack_arrays(array_list, ref_band, time_handling, timestamp=None):
    """Given a list of arrays, stack them to prepare for merging into single dataset 
    or other computations. Note that this function should be used on bands of a single
    scene or scenes from the same tile (i.e. where the geographic area is the same).
    
    Parameters: 
    array_list - list of xarray DataArrays to be stacked
    """
    # Reproject and resample all bands to match the reference band.
    array_list = [ array_list[i].rio.reproject_match(ref_band, fill_value=-9999) for i in range(len(array_list)) ]

    # Chunk the dataset with dask.
    array_list = [ array_list[i].chunk({'x': 1000, 'y': 1000}) for i in range(len(array_list)) ]

    # Assign the same timestamp to each of the bands in the scene.
    if time_handling == 'Bands':
        array_list = [array_list[i].assign_coords({
        "x": array_list[0].x,
        "y": array_list[0].y,
        "time": xr.DataArray([timestamp], dims='time')}) for i in range(len(array_list))]
    
    # Create new time dimension that will vary for each scene.
    elif time_handling == 'Scenes':
        array_list = [array_list[i].assign_coords({
        "x": array_list[0].x,
        "y": array_list[0].y,
        "time": array_list[i].time}) for i in range(len(array_list))]
    
    # Composites should no longer have timestamps associated with them.
    elif time_handling == 'Composites':
        array_list = [array_list[i].assign_coords({
        "x": array_list[0].x,
        "y": array_list[0].y}) for i in range(len(array_list))]

    # Align the arrays to the first array in the list excluding time dimension so it is not overwritten.
    array_list = [ xr.align(array_list[0], array_list[i], join="override", exclude='time', fill_value=-9999)[1] for i in range(len(array_list)) ]

    return array_list


def stack_bands(bands_data, band_nms, timestamp):
    """
    (Originally from genprepWater.py)

    Use one of the bands as a reference for reprojection and resampling (matching the resolution of
    the reference band), then align all bands before merging into a single xarray dataset. 

    Note the original function used the first band by default as the reference band for reprojecting 
    and resampling, which works for Landsat, but for Sentinel-2, the first band has a 10m resolution 
    which caused capacity issues when running, so switched to using bands_data[6] (20m resolution) as 
    the reference band. 

    Parameters:
    bands_data - list of xarray DataArrays for each band
    band_nms- list of strings of the band names
    satellite -  string denoting the satellite (e.g. LANDSAT_8, SENTINEL_2)

    Returns:
    bands_data - xarray dataset containing all bands
    """
    
    # Name the bands so they appear as named data variables in the xarray dataset
    bands_data = [ rename_bands(band_data, band_nms, i) for i,band_data in enumerate(bands_data) ] 

    # Create new time dimension.
    bands_data = [bands_data[i].expand_dims(dim='time') for i in range(len(band_nms))]
   
    # Assign the first band of each scene to be the reference band used when stacking.
    ref_band = bands_data[0]

    bands_data = [ bands_data[i].drop('band').squeeze('band') for i in range(len(band_nms))]

    # Add fill value attribute to -9999
    bands_data = [ bands_data[i].assign_attrs({'_FillValue': -9999}) for i in range(len(band_nms)) ]

    # Stack the bands for each scene. 
    bands_data = stack_arrays(bands_data, ref_band, time_handling = 'Bands', timestamp=timestamp)

    bands_data = xr.merge(bands_data, fill_value=-9999)
    
    # Add attributes from the original reference band
    attrs = ref_band.attrs
    bands_data = bands_data.assign_attrs(attrs)

    return bands_data


def stack_scenes(array_list):
    """
    Prepare to concatenate all the scenes together into a single xarray dataset. 
    
    The input is a list of xarray arrays, each containing a single scene's band data
    for the same tile. The arrays are first aligned to the first array in the list, 
    then the coordinates are copied across all the scenes so xarray is able to align 
    and concatenate properly without dealing with slight differences in coordinates due
    to floats. 
    """

    # Stack the arrays
    array_list_aligned = stack_arrays(array_list, array_list[0], time_handling='Scenes')

    # Turn any 0s (nodata) into -9999s
    array_list_aligned = [ array_list_aligned[i].where(array_list_aligned[i] != 0, -9999) for i in range(len(array_list_aligned)) ]

    # Concatenate the xarray datasets inside array_list into a single xarray dataset
    ds = xr.concat(array_list_aligned, dim='time', fill_value=-9999)

    ds = ds.assign_attrs({'_FillValue': -9999})

    return ds


def ls_water_mask(dataset):
    """Generate water mask for landsat using pixel_qa band."""

    # Select the band to use for water masking (pixel_qa band for Landsat)
    mask_band = (dataset.pixel_qa).astype('int16').assign_attrs({'_FillValue': -9999})
    mask_band = mask_band.where(mask_band != 1, -9999, drop=False)

    # Create a mask for nodata areas around edges of the image (value of 1 in pixel_qa band)
    nodata_mask = (mask_band == -9999)

    # Create a mask for water using the 7th bit
    boolean_mask = ((mask_band & 0b0000000010000000) != 0)
    boolean_mask = boolean_mask & ~nodata_mask

    return boolean_mask

def ls_cloud_mask(dataset):
    """Generate cloud mask for landsat using pixel_qa band."""
    
    # Select the band to use for cloud masking (pixel_qa band for Landsat)
    mask_band = (dataset.pixel_qa).astype('int16').assign_attrs({'_FillValue': -9999})
    mask_band = mask_band.where(mask_band != 1, -9999, drop=False)
    
    # Create a mask for nodata areas around edges of the image (value of 1 in pixel_qa band)
    nodata_mask = (mask_band == -9999)

    # If dilated cloud (1), cloud (3), cloud shadow (4) and snow (5) bits are OFF, pixel is clear
    dilated_cloud_bit = 1 << 1
    cloud_bit = 1 << 3
    cloud_shadow_bit = 1 << 4
    snow_bit = 1 << 5
    clear_bit_mask = dilated_cloud_bit | cloud_bit | cloud_shadow_bit | snow_bit
    boolean_mask = np.bitwise_and(mask_band, clear_bit_mask) == 0

    return boolean_mask

def ls_clean_mask(dataset, keep_water=False):
    # Create the individual water and cloud masks
    water_mask = ls_water_mask(dataset)
    cloud_mask = ls_cloud_mask(dataset)

    if keep_water == False:
        # Combine the water and cloud masks (inverting water so that non-water areas are True)
        water_mask = ~water_mask
        combined_mask = water_mask.where(cloud_mask)
    if keep_water == True:
        # Combine the water and cloud masks (not inverting water so that water areas are True)
        combined_mask = water_mask.where(cloud_mask)

    return combined_mask
    