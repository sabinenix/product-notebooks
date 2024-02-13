import numpy as np
import xarray as xr
import scipy.optimize as opt  #nnls
import os
import collections




# Fractional Cover Function from: https://github.com/SatelliteApplicationsCatapult/datacube-utilities
# Author: KMF
# Creation date: 2016-10-24



csv_file_path = os.path.join(os.path.dirname(__file__), 'AncillaryData/endmembers_landsat.csv')


def frac_coverage_classify(dataset_in, clean_mask=None, no_data=-9999):
    """
    Description:
      Performs fractional coverage algorithm on given dataset. If no clean mask is given, the 'cf_mask'
      variable must be included in the input dataset, as it will be used to create a
      clean mask
    Assumption:
      - The implemented algqorithm is defined for Landsat 5/Landsat 7; in order for it to
        be used for Landsat 8, the bands will need to be adjusted
    References:
      - Guerschman, Juan P., et al. "Assessing the effects of site heterogeneity and soil
        properties when unmixing photosynthetic vegetation, non-photosynthetic vegetation
        and bare soil fractions from Landsat and MODIS data." Remote Sensing of Environment
        161 (2015): 12-26.
    -----
    Inputs:
      dataset_in (xarray.Dataset) - dataset retrieved from the Data Cube (can be a derived
        product, such as a cloudfree mosaic; should contain
          coordinates: latitude, longitude
          variables: blue, green, red, nir, swir1, swir2
        If user does not provide a clean_mask, dataset_in must also include the cf_mask
        variable
    Optional Inputs:
      clean_mask (nd numpy array with dtype boolean) - true for values user considers clean;
        If none is provided, one will be created which considers all values to be clean.
    Output:
      dataset_out (xarray.Dataset) - fractional coverage results with no data = -9999; containing
          coordinates: latitude, longitude
          variables: bs, pv, npv
        where bs -> bare soil, pv -> photosynthetic vegetation, npv -> non-photosynthetic vegetation
    """
    print(csv_file_path)

    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)

    band_stack = []

    mosaic_clean_mask = clean_mask.flatten()

    for band in [
            dataset_in.blue.values, dataset_in.green.values, dataset_in.red.values, dataset_in.nir.values,
            dataset_in.swir1.values, dataset_in.swir2.values
    ]:
        band = band.astype(np.float32)
        #band = band * 0.0001
        band = band.flatten()
        band_clean = np.full(band.shape, np.nan)
        band_clean[mosaic_clean_mask] = band[mosaic_clean_mask]
        band_stack.append(band_clean)

    band_stack = np.array(band_stack).transpose()

    for b in range(6):
        band_stack = np.hstack((band_stack, np.expand_dims(np.log(band_stack[:, b]), axis=1)))
    for b in range(6):
        band_stack = np.hstack(
            (band_stack, np.expand_dims(np.multiply(band_stack[:, b], band_stack[:, b + 6]), axis=1)))
    for b in range(6):
        for b2 in range(b + 1, 6):
            band_stack = np.hstack(
                (band_stack, np.expand_dims(np.multiply(band_stack[:, b], band_stack[:, b2]), axis=1)))
    for b in range(6):
        for b2 in range(b + 1, 6):
            band_stack = np.hstack(
                (band_stack, np.expand_dims(np.multiply(band_stack[:, b + 6], band_stack[:, b2 + 6]), axis=1)))
    for b in range(6):
        for b2 in range(b + 1, 6):
            band_stack = np.hstack((band_stack, np.expand_dims(
                np.divide(band_stack[:, b2] - band_stack[:, b], band_stack[:, b2] + band_stack[:, b]), axis=1)))

    band_stack = np.nan_to_num(band_stack)  # Now a n x 63 matrix (assuming one acquisition)

    ones = np.ones(band_stack.shape[0])
    ones = ones.reshape(ones.shape[0], 1)
    band_stack = np.concatenate((band_stack, ones), axis=1)  # Now a n x 64 matrix (assuming one acquisition)

    end_members = np.loadtxt(csv_file_path, delimiter=',')  # Creates a 64 x 3 matrix

    SumToOneWeight = 0.02
    ones = np.ones(end_members.shape[1]) * SumToOneWeight
    ones = ones.reshape(1, end_members.shape[1])
    end_members = np.concatenate((end_members, ones), axis=0).astype(np.float32)

    result = np.zeros((band_stack.shape[0], end_members.shape[1]), dtype=np.float32)  # Creates an n x 3 matrix

    for i in range(band_stack.shape[0]):
        if mosaic_clean_mask[i]:
            result[i, :] = (opt.nnls(end_members, band_stack[i, :])[0].clip(0, 2.54) * 100).astype(np.int16)
        else:
            result[i, :] = np.ones((end_members.shape[1]), dtype=np.int16) * (-9999)  # Set as no data

    latitude = dataset_in.latitude
    longitude = dataset_in.longitude

    result = result.reshape(latitude.size, longitude.size, 3)

    pv_band = result[:, :, 0]
    npv_band = result[:, :, 1]
    bs_band = result[:, :, 2]

    #pv_clean = np.full(pv_band.shape, -9999)
    #npv_clean = np.full(npv_band.shape, -9999)
    #bs_clean = np.full(bs_band.shape, -9999)
    #pv_clean[clean_mask] = pv_band[clean_mask]
    #npv_clean[clean_mask] = npv_band[clean_mask]
    #bs_clean[clean_mask] = bs_band[clean_mask]

    rapp_bands = collections.OrderedDict([('bs', (['latitude', 'longitude'], bs_band)),
                                          ('pv', (['latitude', 'longitude'], pv_band)),
                                          ('npv', (['latitude', 'longitude'], npv_band))])

    rapp_dataset = xr.Dataset(rapp_bands, coords={'latitude': latitude, 'longitude': longitude})

    return rapp_dataset



def create_default_clean_mask(dataset_in):
    """
    Description:
        Creates a data mask that masks nothing.
    -----
    Inputs:
        dataset_in (xarray.Dataset) - dataset retrieved from the Data Cube.
    Throws:
        ValueError - if dataset_in is an empty xarray.Dataset.
    """
    data_vars = dataset_in.data_vars
    if data_vars:
        first_data_var = next(iter(data_vars))
        clean_mask = np.ones(dataset_in[first_data_var].shape).astype(bool)
        return clean_mask
    else:
        raise ValueError('`dataset_in` has no data!')
    


# Import required packages
import fiona
import collections
import numpy as np
import xarray as xr
from osgeo import osr
from osgeo import ogr
import geopandas as gpd
import rasterio.features
import scipy.interpolate
import multiprocessing as mp
from scipy import ndimage as nd
from skimage.measure import label
from skimage.measure import find_contours
#from geopy.geocoders import Nominatim
from shapely.geometry import mapping, shape
#from datacube.utils.cog import write_cog
#from datacube.helpers import write_geotiff
#from datacube.utils.geometry import assign_crs
#from datacube.utils.geometry import CRS, Geometry
from shapely.geometry import LineString, MultiLineString, shape

def subpixel_contours(da,
                      z_values=[0.0],
                      crs=None,
                      affine=None,
                      attribute_df=None,
                      output_path=None,
                      min_vertices=2,
                      dim='time',
                      errors='ignore',
                      verbose=False):
    
    """
    Uses `skimage.measure.find_contours` to extract multiple z-value 
    contour lines from a two-dimensional array (e.g. multiple elevations
    from a single DEM), or one z-value for each array along a specified 
    dimension of a multi-dimensional array (e.g. to map waterlines 
    across time by extracting a 0 NDWI contour from each individual 
    timestep in an xarray timeseries).    
    
    Contours are returned as a geopandas.GeoDataFrame with one row per 
    z-value or one row per array along a specified dimension. The 
    `attribute_df` parameter can be used to pass custom attributes 
    to the output contour features.
    
    Last modified: November 2020
    
    Parameters
    ----------  
    da : xarray DataArray
        A two-dimensional or multi-dimensional array from which 
        contours are extracted. If a two-dimensional array is provided, 
        the analysis will run in 'single array, multiple z-values' mode 
        which allows you to specify multiple `z_values` to be extracted.
        If a multi-dimensional array is provided, the analysis will run 
        in 'single z-value, multiple arrays' mode allowing you to 
        extract contours for each array along the dimension specified 
        by the `dim` parameter.  
    z_values : int, float or list of ints, floats
        An individual z-value or list of multiple z-values to extract 
        from the array. If operating in 'single z-value, multiple 
        arrays' mode specify only a single z-value.
    crs : string or CRS object, optional
        An EPSG string giving the coordinate system of the array 
        (e.g. 'EPSG:3577'). If none is provided, the function will 
        attempt to extract a CRS from the xarray object's `crs` 
        attribute.
    affine : affine.Affine object, optional
        An affine.Affine object (e.g. `from affine import Affine; 
        Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "6886890.0) giving the 
        affine transformation used to convert raster coordinates 
        (e.g. [0, 0]) to geographic coordinates. If none is provided, 
        the function will attempt to obtain an affine transformation 
        from the xarray object (e.g. either at `da.transform` or
        `da.geobox.transform`).
    output_path : string, optional
        The path and filename for the output shapefile.
    attribute_df : pandas.Dataframe, optional
        A pandas.Dataframe containing attributes to pass to the output
        contour features. The dataframe must contain either the same 
        number of rows as supplied `z_values` (in 'multiple z-value, 
        single array' mode), or the same number of rows as the number 
        of arrays along the `dim` dimension ('single z-value, multiple 
        arrays mode').
    min_vertices : int, optional
        The minimum number of vertices required for a contour to be 
        extracted. The default (and minimum) value is 2, which is the 
        smallest number required to produce a contour line (i.e. a start
        and end point). Higher values remove smaller contours, 
        potentially removing noise from the output dataset.
    dim : string, optional
        The name of the dimension along which to extract contours when 
        operating in 'single z-value, multiple arrays' mode. The default
        is 'time', which extracts contours for each array along the time
        dimension.
    errors : string, optional
        If 'raise', then any failed contours will raise an exception.
        If 'ignore' (the default), a list of failed contours will be
        printed. If no contours are returned, an exception will always
        be raised.
    verbose : bool, optional
        Print debugging messages. Default False.
        
    Returns
    -------
    output_gdf : geopandas geodataframe
        A geopandas geodataframe object with one feature per z-value 
        ('single array, multiple z-values' mode), or one row per array 
        along the dimension specified by the `dim` parameter ('single 
        z-value, multiple arrays' mode). If `attribute_df` was 
        provided, these values will be included in the shapefile's 
        attribute table.
    """

    def contours_to_multiline(da_i, z_value, min_vertices=2):
        '''
        Helper function to apply marching squares contour extraction
        to an array and return a data as a shapely MultiLineString.
        The `min_vertices` parameter allows you to drop small contours 
        with less than X vertices.
        '''
        
        # Extracts contours from array, and converts each discrete
        # contour into a Shapely LineString feature. If the function 
        # returns a KeyError, this may be due to an unresolved issue in
        # scikit-image: https://github.com/scikit-image/scikit-image/issues/4830
        line_features = [LineString(i[:,[1, 0]]) 
                         for i in find_contours(da_i.data, z_value) 
                         if i.shape[0] > min_vertices]        

        # Output resulting lines into a single combined MultiLineString
        return MultiLineString(line_features)

    # Check if CRS is provided as a xarray.DataArray attribute.
    # If not, require supplied CRS
    try:
        crs = da.crs
    except:
        if crs is None:
            raise ValueError("Please add a `crs` attribute to the "
                             "xarray.DataArray, or provide a CRS using the "
                             "function's `crs` parameter (e.g. 'EPSG:3577')")

    # Check if Affine transform is provided as a xarray.DataArray method.
    # If not, require supplied Affine
    try:
        affine = da.geobox.transform
    except KeyError:
        affine = da.transform
    except:
        if affine is None:
            raise TypeError("Please provide an Affine object using the "
                            "`affine` parameter (e.g. `from affine import "
                            "Affine; Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "
                            "6886890.0)`")

    # If z_values is supplied is not a list, convert to list:
    z_values = z_values if (isinstance(z_values, list) or 
                            isinstance(z_values, np.ndarray)) else [z_values]

    # Test number of dimensions in supplied data array
    if len(da.shape) == 2:
        if verbose:
            print(f'Operating in multiple z-value, single array mode')
        dim = 'z_value'
        contour_arrays = {str(i)[0:10]: 
                          contours_to_multiline(da, i, min_vertices) 
                          for i in z_values}    

    else:

        # Test if only a single z-value is given when operating in 
        # single z-value, multiple arrays mode
        if verbose:
            print(f'Operating in single z-value, multiple arrays mode')
        if len(z_values) > 1:
            raise ValueError('Please provide a single z-value when operating '
                             'in single z-value, multiple arrays mode')

        contour_arrays = {str(i)[0:10]: 
                          contours_to_multiline(da_i, z_values[0], min_vertices) 
                          for i, da_i in da.groupby(dim)}

    # If attributes are provided, add the contour keys to that dataframe
    if attribute_df is not None:

        try:
            attribute_df.insert(0, dim, contour_arrays.keys())
        except ValueError:

            raise ValueError("One of the following issues occured:\n\n"
                             "1) `attribute_df` contains a different number of "
                             "rows than the number of supplied `z_values` ("
                             "'multiple z-value, single array mode')\n"
                             "2) `attribute_df` contains a different number of "
                             "rows than the number of arrays along the `dim` "
                             "dimension ('single z-value, multiple arrays mode')")

    # Otherwise, use the contour keys as the only main attributes
    else:
        attribute_df = list(contour_arrays.keys())

    # Convert output contours to a geopandas.GeoDataFrame
    contours_gdf = gpd.GeoDataFrame(data=attribute_df, 
                                    geometry=list(contour_arrays.values()),
                                    crs=crs)   

    # Define affine and use to convert array coords to geographic coords.
    # We need to add 0.5 x pixel size to the x and y to obtain the centre 
    # point of our pixels, rather than the top-left corner
    shapely_affine = [affine.a, affine.b, affine.d, affine.e, 
                      affine.xoff + affine.a / 2.0, 
                      affine.yoff + affine.e / 2.0]
    contours_gdf['geometry'] = contours_gdf.affine_transform(shapely_affine)

    # Rename the data column to match the dimension
    contours_gdf = contours_gdf.rename({0: dim}, axis=1)

    # Drop empty timesteps
    empty_contours = contours_gdf.geometry.is_empty
    failed = ', '.join(map(str, contours_gdf[empty_contours][dim].to_list()))
    contours_gdf = contours_gdf[~empty_contours]

    # Raise exception if no data is returned, or if any contours fail
    # when `errors='raise'. Otherwise, print failed contours
    if empty_contours.all() and errors == 'raise':
        raise RuntimeError("Failed to generate any valid contours; verify that "
                           "values passed to `z_values` are valid and present "
                           "in `da`")
    elif empty_contours.all() and errors == 'ignore':
        if verbose:
            print ("Failed to generate any valid contours; verify that "
                    "values passed to `z_values` are valid and present "
                    "in `da`")
    elif empty_contours.any() and errors == 'raise':
        raise Exception(f'Failed to generate contours: {failed}')
    elif empty_contours.any() and errors == 'ignore':
        if verbose:
            print(f'Failed to generate contours: {failed}')

    # If asked to write out file, test if geojson or shapefile
    if output_path and output_path.endswith('.geojson'):
        if verbose:
            print(f'Writing contours to {output_path}')
        contours_gdf.to_crs('EPSG:4326').to_file(filename=output_path, 
                                                 driver='GeoJSON')

    if output_path and output_path.endswith('.shp'):
        if verbose:
            print(f'Writing contours to {output_path}')
        contours_gdf.to_file(filename=output_path)
        
    return contours_gdf