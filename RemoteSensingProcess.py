### This python script is intended to hold self-defined functions to process data used in analyzing the fractional irrigated area coverage

### import relevant libraries
# reading files
import os
import rasterio 
import geopandas as gpd
import pandas as pd

# processing raster
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import datetime as dt

# visualize raster
import matplotlib.pyplot as plt


### function for calculating maximum and minimum NDVI across the specified year per timeseries
def NDVI_minmax(pd_df, code, start = 2000, end = 2019, indiv = False):
    '''
    Calculate the maximum and minimum NDVI (Normalized Difference Vegetation Index) across specified years.

    Parameters:
    pd_df (DataFrame): A pandas DataFrame containing NDVI data.
    code (str): The column name representing the pixel ID.
    start (int): The start year for the analysis (inclusive).
    end (int): The end year for the analysis (exclusive).
    indiv (bool): If True, return individual seasonal NDVI values. Default is False.

    Returns:
    DataFrame: A DataFrame containing the modified date, maximum NDVI, and minimum NDVI.
    DataFrame (optional): A DataFrame containing individual seasonal NDVI values if indiv is True.
    '''

    # set up specified pixel ID column
    code_df = pd_df.loc[:,['modified date', code]]

    # Initialize empty lists to store seasonal NDVI and date information
    seasonal_NDVI = []
    seasonal_date = []
    
    # set up selected year range
    year_array = np.arange(start, end)

    # iterate over selected year 
    for year in year_array:
        start = year
        end = year + 1
        
        # Create a mask to filter data for the current year's NDVI values
        seasonal_mask = (code_df['modified date'] >= f'{start}-11-01') & (code_df['modified date'] <= f'{end}-10-31')    

        # Extract NDVI values for the current year and append them to the list
        NDVI = code_df[seasonal_mask][code]
        seasonal_NDVI.append(NDVI)
        
        # Append the date range to the list
        seasonal_date.append(f'{start}-{end}')

    # stack NDVI into array of shape (year, NDVI values from Nov - Oct)
    code_stack = np.stack(seasonal_NDVI,axis = 0)

    # compute maximum and minimum from each year for each of the 23 date
    max_NDVI = list(np.nanmax(code_stack, axis = 0))
    min_NDVI = list(np.nanmin(code_stack, axis = 0))

    # set up display date to show month and day
    seasonal_display_date = code_df[seasonal_mask]['modified date']
    seasonal_display_date = seasonal_display_date.apply(lambda x: x.strftime('%m-%d'))

    # set up df for output
    df = pd.DataFrame({
        'modified date': seasonal_display_date,
        'max NDVI': max_NDVI,
        'min NDVI': min_NDVI
    })
    # Reset the index to start from 0
    df.reset_index(drop=True, inplace=True)

    # optional output for individual season used in creating max and min NDVI
    if indiv == True:
        indiv_NDVI_dict = {
            'modified date': seasonal_display_date
        }
        
        for index, season in enumerate(seasonal_date):
            indiv_NDVI_dict[season] = list(seasonal_NDVI[index])
        
        df_indiv = pd.DataFrame(indiv_NDVI_dict)

        # Reset the index of df_indiv as well if needed
        df_indiv.reset_index(drop=True, inplace=True)

        return df, df_indiv

    else:
        return df

### function for calculating longitude and latitude geographic coordinates from pixel coordinate of a binary tif np array
def pixel_geo_coord(tif_path, TF_array):
    """
    Converts pixel coordinates to geographic coordinates for a given TIFF image.

    Args:
    tif_path (str): Path to the TIFF image file.
    TF_array (numpy.ndarray): A binary array (True/False) indicating the pixels to be converted.

    Returns:
    numpy.ndarray: A NumPy array containing the geographic coordinates (longitude, latitude)
    corresponding to the True pixels in the input TF_array.

    This function opens a TIFF image using rasterio, extracts the geographic coordinates
    of the True pixels in the input TF_array, and returns the coordinates in a NumPy array.

    Example:
    tif_path = "path/to/your/tif_image.tif"
    TF_array = np.array([[False, True, False],
                        [True, False, True]])
    coordinates = pixel_geo_coord(tif_path, TF_array)
    """
    # open tif image to enable coordinate readings
    with rasterio.open(tif_path) as coord_conversion_img:
        
        # acquire True pixel coordinates
        True_pixel_row_indices, True_pixel_col_indices = np.where(TF_array)

        # Convert pixel coordinates to geographic coordinates
        True_pixel_lon_values, True_pixel_lat_values = coord_conversion_img.xy(True_pixel_row_indices, True_pixel_col_indices)

        # Combine lon and lat values into a NumPy array
        True_pixel_geo_coord_array = np.column_stack((True_pixel_lon_values, True_pixel_lat_values))

        return True_pixel_geo_coord_array
    
### function for calculating pixel row column coordinates corresponding to the input longitude and latitude geographic coordinates 
def geo_pixel_coord(tif_path, lon_lat_array, coord_count = False):
    """
    Converts geographic coordinates to pixel coordinates within a TIFF image.

    Args:
    tif_path (str): Path to the TIFF image file.
    lon_lat_array (numpy.ndarray): An array containing pairs of longitude and latitude values.
    coord_count (bool, optional): If True, returns both unique pixel coordinates and their counts.
                                   If False, returns only unique pixel coordinates. Default is False.

    Returns:
    tuple or numpy.ndarray: Depending on the value of coord_count, either a tuple containing
    a NumPy array of unique pixel coordinates and a NumPy array of counts (if coord_count=True),
    or just a NumPy array of unique pixel coordinates.

    This function opens a TIFF image using rasterio, converts the provided geographic coordinates
    into pixel coordinates, and returns either the unique pixel coordinates or both the unique
    pixel coordinates and their counts.

    Args:
    - tif_path (str): The path to the TIFF image to which the conversion is performed.
    - lon_lat_array (numpy.ndarray): A 2D array containing pairs of geographic coordinates.
    - coord_count (bool, optional): If True, both unique pixel coordinates and their counts are returned.
      If False, only unique pixel coordinates are returned. Default is False.

    Returns:
    - If coord_count is False, it returns a NumPy array of unique pixel coordinates.
    - If coord_count is True, it returns a tuple with two elements:
      - The first element is a NumPy array of unique pixel coordinates.
      - The second element is a NumPy array of counts, indicating how many times each unique pixel coordinate appears in the input.

    This function utilizes the rasterio library to perform coordinate transformation, converting the provided geographic coordinates
    into pixel coordinates within the specified TIFF image. It can be used to align spatial data and perform operations on the
    corresponding pixel locations within the image.

    Example:
    tif_path = "path/to/your/tif_image.tif"
    lon_lat_array = np.array([[longitude1, latitude1],
                             [longitude2, latitude2]])
    unique_coords = geo_pixel_coord(tif_path, lon_lat_array)
    unique_coords, counts = geo_pixel_coord(tif_path, lon_lat_array, coord_count=True)
    """
    with rasterio.open(tif_path) as MODIS_img:
        # convert LULC geo coord into MODIS pixel coord
        MODIS_row_indices, MODIS_col_indices = MODIS_img.index(lon_lat_array[:, 0], lon_lat_array[:, 1]) # input lon, lat
        # pair cooresponding row and pixel pair into np array
        MODIS_pixel_coord_array = np.column_stack((MODIS_row_indices, MODIS_col_indices))
        # get unique MODIS pixel coord
        unique_MODIS_pixel_coord_array, counts = np.unique(MODIS_pixel_coord_array, return_counts = True, axis=0)

        return unique_MODIS_pixel_coord_array, counts

### function to create reference timeseries from pixel coordinate mask 
def reference_timeseries(reference_pixel_mask, NDVI_timeseries_array):

    """
    Calculate a reference time series from a pixel coordinate mask and the corresponding NDVI time series data.

    Args:
    reference_pixel_mask (numpy.ndarray): A binary mask array of 1 and np.nan indicating reference pixel locations.
    NDVI_timeseries_array (numpy.ndarray): A multi-band NDVI time series data array.

    Returns:
    dict: A dictionary containing upper, mean, lower, and std reference time series.

    This function creates a reference time series by selecting pixels based on a reference pixel mask and calculating
    the average (mean) and standard deviation (std) of NDVI values for each timestep. It also computes the upper and lower
    thresholds (mean ± std) for the reference time series.

    Example:
    reference_mask = np.array([[True, False, False],
                              [False, True, True]])
    NDVI_data = np.array([[0.5, 0.6, 0.7],
                         [0.4, 0.7, 0.8]])
    reference_series = reference_timeseries(reference_mask, NDVI_data)
    """
    
    # duplicate mask stack to match the number of timestep bands in the NDVI raster
    expanded_reference_pixel_mask = np.tile(reference_pixel_mask, (NDVI_timeseries_array.shape[0], 1, 1))
    # select only reference pixels
    reference_NDVI_timeseries_array = expanded_reference_pixel_mask * NDVI_timeseries_array

    # average the NDVI values for each timestep creating average reference timeseries
    consolidated_reference_mean = np.nanmean(reference_NDVI_timeseries_array, axis = (1,2))
    # calculate standard deviation for the NDVI values for each timestep
    consolidated_reference_std = np.nanstd(reference_NDVI_timeseries_array, axis = (1,2))

    # average + std
    consolidated_reference_std_upper = consolidated_reference_mean + consolidated_reference_std

    # average - std
    consolidated_reference_std_lower = consolidated_reference_mean - consolidated_reference_std
    
    # Combine mean and standard deviation into a reference time series array
    reference_time_series = {'upper': consolidated_reference_std_upper, 'mean': consolidated_reference_mean, 
                             'lower': consolidated_reference_std_lower, 'std': consolidated_reference_std}

    return reference_time_series

### function to condolidate reference timeseries into a single reference threshold number 
def reference_threshold(reference_mean_timeseries, reference_std_timeseries):
  
    """
    Calculate reference thresholds based on the mean and standard deviation of reference time series.

    Args:
    reference_mean_timeseries (numpy.ndarray): Array containing the mean values of the reference time series.
    reference_std_timeseries (numpy.ndarray): Array containing the standard deviation values of the reference time series.

    Returns:
    dict: A dictionary containing upper, mean, lower, and std thresholds.

    This function calculates reference thresholds by combining the mean and standard deviation of reference time series data.
    It computes the upper threshold (mean + std), lower threshold (mean - std), and the mean threshold. These thresholds
    provide a range of values for evaluating data points against a reference.

    Example:
    reference_mean = np.array([0.6, 0.7, 0.8])
    reference_std = np.array([0.1, 0.15, 0.12])
    thresholds = reference_threshold(reference_mean, reference_std)
    """

    # average the reference std for all timeseteps into one number
    std_threshold = np.mean(reference_std_timeseries)
    # average the reference mean for all timeseteps into one number
    average_threshold = np.mean(reference_mean_timeseries)
    # mean + std
    upper_threshold = average_threshold + std_threshold
    # mean - std
    lower_threshold = average_threshold - std_threshold

    # consolidate the threshold data
    threshold_data_dict = {'upper': upper_threshold, 'mean': average_threshold,
                           'lower':lower_threshold, 'std': std_threshold}

    return threshold_data_dict

### Output numerical value for cooresponding input LULC type
def LULC_key(lulc_type):
    '''
    Map a land use and land cover (LULC) type to its corresponding numerical value.
    
    Args:
    lulc_type (str): The LULC type to be mapped.

    Returns:
    int: The numerical value corresponding to the given LULC type.
    
    Raises:
    KeyError: If the provided LULC type is not found in the dictionary.

    Example:
    >>> LULC_key('trees')
    1
    '''

    # Define a dictionary mapping LULC types to numerical values
    lulc_dict = {
        'water': 0, 'trees': 1, 'grass': 2, 'flooded_vegetation': 3,
        'crops': 4, 'shrub_and_scrub': 5, 'built': 6, 'bare': 7,
        'snow_and_ice': 8
    }

    try:
        return lulc_dict[lulc_type]
    except KeyError:
        raise KeyError(f"LULC type '{lulc_type}' not found in the dictionary.")

### Output geo coordinate from LULC pixel coordinate
def LULCstack_to_LULC_coord(lulc_type_stack_filepath, lulc_type, notification = False):
    
    '''
    Extracts the geographical coordinates (latitude and longitude) of pixels corresponding to a specific Land Use/Land Cover (LULC) type from a LULC stack.

    Args:
        lulc_type_stack_filepath (str): The file path of the LULC stack.
        lulc_type (str): The specific LULC type to extract coordinates for.
        notification (bool, optional): Whether to enable progress notifications. Default is False.

    Returns:
        dict: A dictionary where keys are timestamps and values are arrays of geographical coordinates for the specified LULC type.
    '''

    ### Match geo coord to modis pixel coord

    # initialize variable to hold LULC geo coord at different timesteps
    LULC_geo_coord_dict = {}

    # read LULC type stack  
    with rasterio.open(lulc_type_stack_filepath) as LULC_type_img:
        # read date
        LULC_type_stack_date = LULC_type_img.descriptions # each item is date value

        # optional: turn on notification
        if notification:
            # Calculate progress step
            total_timesteps = len(LULC_type_stack_date)  # Calculate the total number of timesteps
            progress_steps = [0.1, 0.25, 0.50, 0.75]
            current_progress_step = 0

        # iterate through each date to extract band and LULC geo coord from pixel coord
        for index, date in enumerate(LULC_type_stack_date):
            
            # Read band
            LULC_type_stack_band = LULC_type_img.read(index+1)

            # Convert band into True False with True representing desired LULC type pixels
            date_mask = (LULC_type_stack_band) == LULC_key(lulc_type)
            
            # Convert desired pixels' rows and columns into longitude and latitude
            LULC_geo_coord_array = pixel_geo_coord(lulc_type_stack_filepath, date_mask)

            # store LULC geo coordinate in dictionary
            LULC_geo_coord_dict[date] = LULC_geo_coord_array
    
            # optional: Check if the current progress is at one of the specified steps (10%, 25%, 50%, 75%)
            if notification:
                # Calculate progress
                percent = (index + 1) / total_timesteps 
                if current_progress_step < len(progress_steps) and percent >= progress_steps[current_progress_step]:
                    print(f"Pixel-Geo coordinate conversion progress: {int(progress_steps[current_progress_step] * 100)}%")
                    current_progress_step += 1

    return LULC_geo_coord_dict

### Output modis LULC pixel count array from dict
def LULC_coordDict_MODIS_count(MODIS_filepath, LULC_geo_coord_dict, notification = False):
    
    '''
    Calculate MODIS count maps for each date within the dictionary by matching MODIS pixel coordinates with LULC geo coordinates.

    Args:
    MODIS_filepath (str): File path to the MODIS data.
    LULC_geo_coord_dict (dict): A dictionary where keys are dates and values are arrays of geographical coordinates.
    notification (bool, optional): If True, print progress information during the calculation. Default is False.

    Returns:
    np.ndarray: A 3D array containing MODIS count maps for each date.

    Example:
    ### counts = calculate_MODIS_LULC_counts('MODIS.tif', LULC_geo_coord_dict, notification=True)
    ### counts.shape  # Shape of the output array
    (5, 482, 493)
    '''
    
    # initialize corresponding raster with equal bands to LULC image to hold count
    MODIS_stacked_LULCtype_count = np.empty((len(LULC_geo_coord_dict.keys()), 482, 493), dtype=np.float32)
    
    # optional: set up for notifying progress
    if notification:
    # Calculate progress step
        total_timesteps = len(LULC_geo_coord_dict)  # Calculate the total number of timesteps
        index_notif = 0
        progress_steps = [0.1, 0.25, 0.50, 0.75]
        current_progress_step = 0

    # iterate through each band in the LULC stack with dictionary
    for index, geo_coord_array in enumerate(LULC_geo_coord_dict.values()):

        # Match MODIS pixels' rows and columns to converted longitude and latitude
        unique_MODIS_pixel_coord_array, counts = geo_pixel_coord(MODIS_filepath,geo_coord_array)

        # Use MODIS row and column to replace band pixel value with corresponding LULC pixel count
        with rasterio.open(MODIS_filepath) as MODIS_img:
            # initialize nan band to hold count
            count_band = np.full(MODIS_img.shape, np.nan, dtype=float)
            
            # loop through each pixel coordinate and count 
            for coord, count in zip(unique_MODIS_pixel_coord_array, counts):
                row, col = coord # extract row and column from coord
                
                # Set the count at the specified row and column in the mask
                count_band[row, col] = count
            # get LULC pixel count that fits into corresponding MODIS pixel
            
            ### replace dummy band for LULC type count band
            MODIS_stacked_LULCtype_count[index, :, :] = count_band
            
            # optional: Check if the current progress is at one of the specified steps (25%, 50%, 75%)
            if notification:
                # Calculate progress
                index_notif += 1
                percent = (index_notif) / total_timesteps 
                if current_progress_step < len(progress_steps) and percent >= progress_steps[current_progress_step]:
                    print(f"Geo-Pixel coordinate conversion progress: {int(progress_steps[current_progress_step] * 100)}%")
                    current_progress_step += 1


    return MODIS_stacked_LULCtype_count
            
### Add 0 count to count array
def add_noCoverage(MODIS_raster_array, single_array_with_nan):

    '''
    Add a 0 count value to MODIS pixels using an existing array with NaN values.

    Args:
    MODIS_raster_array (np.ndarray): The multi-band MODIS raster array.
    single_array_with_nan (np.ndarray): An array with NaN values, indicating no data coverage.

    Returns:
    np.ndarray: The updated MODIS raster array with 0 counts in areas where no data coverage was previously indicated by NaN values.

    Example:
    ### new_MODIS_array = add_noCoverage(MODIS_data, no_data_array)
    '''    
 
    # Duplicate the targeted stack
    no_coverage_array = MODIS_raster_array.copy()
    
    # Create a True/False array where True represents non-white space and False represents NaN white space
    not_whiteSpace_mask = np.where(~np.isnan(single_array_with_nan), True, False) 
    
    # iterate through each band of the targeted stack
    for band_index in np.arange(no_coverage_array.shape[0]):

        # create nan mask where True represents a nan coordinate
        nan_mask = np.isnan(no_coverage_array[band_index])

        # Use the NaN mask and not-white-space mask to identify NaN values that are not in white space areas and implement a 0 count
        no_coverage_array[band_index, nan_mask & not_whiteSpace_mask] = 0

    return no_coverage_array

### Export file array to tif file
def count_raster_export(MODIS_filepath, outdir, raster_array, description_list = 'default'):
    
    '''
    Export a multi-band raster dataset to a specified output file with an updated profile.

    Args:
    MODIS_filepath (str): File path to the existing MODIS dataset.
    outdir (str): File path for the output raster dataset.
    raster_array (np.ndarray): A 3D NumPy array representing the raster data (bands, rows, columns).
    description_list (list, optional): A list of band descriptions. If not provided, default descriptions are used.

    Returns:
    None

    Example:
    ### count_raster_export('MODIS.tif', 'output.tif', my_raster_array, description_list=['Band 1', 'Band 2'])
    '''
    
    # get profile
    with rasterio.open(MODIS_filepath) as MODIS_img:
        MODIS_img_profile = MODIS_img.profile
    
    # update profile
    MODIS_img_profile['nodata'] = np.nan
    MODIS_img_profile['dtype'] = raster_array.dtype
    MODIS_img_profile['count'] = raster_array.shape[0]

    if description_list == 'default':
        description_list = ['no description added'] *  raster_array.shape[0]
    
    # export with updated profile
    with rasterio.open(outdir, 'w', **MODIS_img_profile) as src:
        for index, description in enumerate(description_list):
            src.write(raster_array[index], index + 1)  # Write the data to band i + 1
            # Set the description for the band
            src.set_band_description(index + 1, description)

def band_export(MODIS_filepath, outdir, band_array, description = 'default'):
    
    """
    Export a single-band raster from a NumPy array to a new GeoTIFF file with an updated profile.

    Parameters:
    MODIS_filepath (str): Filepath to an existing GeoTIFF file from which the profile is derived.
    outdir (str): Directory where the new GeoTIFF file will be saved.
    band_array (numpy.ndarray): NumPy array containing the data for the single band to be exported.
    description (str, optional): A description for the band in the output GeoTIFF file. Default is 'default'.

    Returns:
    None

    Note:
    - The function updates the profile of the output GeoTIFF, sets the NoData value to NaN, and defines the data type.
    - If no description is provided, a default description is used.

    Example usage:
    band_export('input.tif', 'output.tif', band_data, 'NDVI Band')
    """

    # get profile
    with rasterio.open(MODIS_filepath) as MODIS_img:
        MODIS_img_profile = MODIS_img.profile
        # update profile
        MODIS_img_profile['nodata'] = np.nan
        MODIS_img_profile['dtype'] = band_array.dtype
        MODIS_img_profile['count'] = 1
    
    if description == 'default':
        description = 'no description added'
    
    # export with updated profile
    with rasterio.open(outdir, 'w', **MODIS_img_profile) as src:
        src.write(band_array, 1)
        src.set_band_description(1, description)

def filter_coverage(count_raster_path, coverage_type, pixel_coverage_number):

    '''
    Filter a count raster stack based on coverage criteria and return a boolean array.

    Args:
        count_raster_path (str): Path to the count array raster.
        coverage_type (str): The type of coverage filter ('more-equal', 'equal', or 'range').
        pixel_coverage_number (int or list): The coverage threshold(s) for filtering.

    Returns:
        count_stack_raster (numpy.ndarray): The filtered count raster as a boolean array.
        count_stack_date (list): List of descriptions corresponding to the date for each band.
    '''

    # read the count array raster
    with rasterio.open(count_raster_path) as count_stack_img:
        count_stack_raster = count_stack_img.read()
        count_stack_date = count_stack_img.descriptions

    ### filter for coverage type and number into True False array ###
    
    if coverage_type == 'more-equal':
        count_stack_raster[count_stack_raster < pixel_coverage_number] = np.nan
        count_stack_raster[count_stack_raster >= pixel_coverage_number] = 1

    elif coverage_type == 'equal':
        count_stack_raster[count_stack_raster != pixel_coverage_number] = np.nan
        count_stack_raster[count_stack_raster == pixel_coverage_number] = 1

    elif coverage_type == 'range':
        if isinstance(pixel_coverage_number, list) and len(pixel_coverage_number) == 2:
            # delineate for more than maximum
            count_stack_raster[count_stack_raster > pixel_coverage_number[0]] = np.nan
            # delineate for less than minimum
            count_stack_raster[count_stack_raster < pixel_coverage_number[1]] = np.nan
            # select for within range
            count_stack_raster[(count_stack_raster <= pixel_coverage_number[0]) &
                               (count_stack_raster >= pixel_coverage_number[1])] = 1
        else:
            raise ValueError("For 'range' coverage type, pixel_coverage_number should be a list of two values.")

    else:
        raise ValueError("Invalid coverage_type. Use 'more-equal', 'equal' or 'range'.")

    return count_stack_raster, count_stack_date

def extract_monthly_coverage(nan_1_stack, nan_1_stack_date, extraction_type, selected_date_keystring_list):

    '''
    Compress monthly dry season coverage arrays into a single array representing the dry season between late 2015 to early 2019.

    Args:
        nan_1_stack (numpy.ndarray): Monthly coverage stack.
        nan_1_stack_date (list): List of date descriptions for each band.
        extraction_type (str): Type of extraction, 'any' to extract if any month in a dry season matches or 'all' to extract if all months in a dry season match.
        selected_date_keystring_list (list): List of keystrings for selecting desired months.

    Returns:
        All_drySeason_compressedCoverage_raster (numpy.ndarray): Compressed coverage raster stack.
        All_drySeason_compressedCoverage_date (list): List of date ranges used to create compressed bands.
    '''

    # initialize list for storing single dryseason array and corresponding dates used to create the single array
    All_drySeason_compressedCoverage_raster = []
    All_drySeason_compressedCoverage_date = []

    # set period
    periods = [
        {'date': ['2015-11', '2015-12', '2016-01', '2016-02', '2016-03', '2016-04'], 'index': [0, 1, 2, 3, 4, 5]},
        {'date': ['2016-11', '2017-01', '2017-02', '2017-03', '2017-04' ], 'index': [6, 7, 8, 9, 10]},
        {'date': ['2017-11','2017-12', '2018-01', '2018-02', '2018-03', '2018-04'], 'index': [11, 12, 13, 14, 15, 16]},
        {'date': ['2018-11', '2018-12', '2019-01', '2019-02', '2019-03', '2019-04'], 'index': [17, 18, 19, 20, 21, 22]}
    ]


    # iterate through each season or less
    for period_info in periods:    
        period_indices = period_info['index']
        period_dates = period_info['date']

    ### Select indices corresponding to specified date in a dry season with period_dates ###
        
        # initialize variable to store indices corresponding to seleccted month in a dry season
        selected_drySeason_indices = []

        # iterate through the date to select corresponding indices
        for month_index, date in enumerate(period_dates):
            if any(keystring in date for keystring in selected_date_keystring_list):
                selected_drySeason_indices.append(period_indices[month_index])

        # order the indices from smallest to largest
        selected_drySeason_indices = sorted(selected_drySeason_indices)

    ### Use the selected indices from earlier to choose which band/date in nan_1_stack ###

         # initialize variable to store the selected band for this dry season
        singleSeason_coverage_bands = []

        # iterate though selected_period_indices
        for selected_period_index in selected_drySeason_indices:
            dry_season_band = nan_1_stack[selected_period_index, :, :]
            singleSeason_coverage_bands.append(dry_season_band)
        
        # Stack the bands for this dry season along the time axis
        singleSeason_coverage_bands = np.stack(singleSeason_coverage_bands, axis = 0)

    ### Compress the coverage bands for this season into a single coverage band ###
        
        if extraction_type == 'any':
            compressed_singleSeason_coverage_band = np.any(singleSeason_coverage_bands == 1, axis = 0).astype(np.float32)
            
        elif extraction_type == 'all':
             compressed_singleSeason_coverage_band = np.all(singleSeason_coverage_bands == 1, axis = 0).astype(np.float32)
        else:
            raise ValueError('Invalid extraction_type parameter. Please input extraction_type parameter as "any" or "all" ')
        compressed_singleSeason_coverage_band[compressed_singleSeason_coverage_band != 1] = np.nan
        
        # store this period's compressed band 
        All_drySeason_compressedCoverage_raster.append(compressed_singleSeason_coverage_band)

        # store the date used to create compressed coverage band
        start_date = nan_1_stack_date[selected_drySeason_indices[0]]
        end_date = nan_1_stack_date[selected_drySeason_indices[-1]]
        period_date_range = f"{start_date} - {end_date}"
        All_drySeason_compressedCoverage_date.append(period_date_range)
    
    # stack the compressed raster list
    All_drySeason_compressedCoverage_raster = np.stack(All_drySeason_compressedCoverage_raster, axis = 0)

    # return raster stack and date list
    return All_drySeason_compressedCoverage_raster, All_drySeason_compressedCoverage_date



def dryMonth_keystring_selection(month_name_list):
    """
    Generate keystrings for dry months based on a list of month names.

    Args:
        month_name_list (list of str): A list of month names, e.g., ['Nov', 'Dec'].

    Returns:
        list of str: A list of keystrings for the selected dry months.
    """
    
    # Dictionary mapping month names to keystrings
    month_number_dict = {
        'Nov': ['10-31', '-11'],
        'Dec': ['-12'],
        'Jan': ['-01'], 
        'Feb': ['-02'],
        'Mar': ['-03'], 
        'Apr': ['-04']
    }
    
    desired_month_keystrings = []
    # Iterate through the input month names
    for month_name in month_name_list:
        try:
            keystrings = month_number_dict[month_name]
            desired_month_keystrings.extend(keystrings)
        except KeyError:
            raise ValueError(f"Invalid month name: {month_name}. Please use valid month names as 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr' ")

    return desired_month_keystrings

def multi_month_coverage_mask_selection(coverageMask_folder, lulc_type, coverage_type):
    
    '''
    Reads and processes a collection of coverage masks for a specific land use and coverage type.
    
    This function scans a folder containing coverage mask files and filters them based on the provided
    land use type and coverage type criteria. It then extracts information from these masks, including
    the array representing the mask and associated date information used to create these masks.

    Parameters:
    - coverageMask_folder (str): The directory path where coverage mask files are stored.
    - lulc_type (str): The specific land use/land cover (LULC) type to filter the masks (e.g., 'trees', 'crops').
    - coverage_type (str): The type of coverage (e.g., 'No' or 'Full') to filter the masks.

    Returns:
    - diffMonthCreation_coverage_mask_dict (dict): A dictionary containing processed coverage masks.
      The keys represent different months or combinations used to create the coverage mask, and each entry
      is a dictionary with 'array' (the mask array) and 'date' (associated date information).

    Example Usage:
    diffMonthCreation_coverage_mask_dict = multi_coverage_mask_selection('path/to/coverage/masks', 'trees', 'No')
    '''


    # Initialize a dictionary to store and output the processed coverage masks
    diffMonthCreation_coverage_mask_dict = {}

    # Acquire a list of coverage mask filenames that match the specified LULC and coverage type
    coverageMask_filenames = [file for file in os.listdir(coverageMask_folder) 
                              if file.endswith('.tif') and 
                              f'{coverage_type.lower()} coverage' in file.lower() and 
                              lulc_type in file]

    # Iterate through the list of filenames to read and process each mask
    for coverageMask_filename in coverageMask_filenames:
        path = os.path.join(coverageMask_folder, coverageMask_filename)

        # Open the coverage mask file using the rasterio library
        with rasterio.open(path) as img:
            coverageMask_array =  img.read()
            coverageMask_date = img.descriptions
        
        # Extract the information regarding selected month or combination used in creating the coverage mask
        selected_month = coverageMask_filename.split()[6] 
        
        # Extract the selected month or combination used in creating the coverage mask
        diffMonthCreation_coverage_mask_dict[selected_month] = {'array': coverageMask_array, 'date': coverageMask_date}

    return diffMonthCreation_coverage_mask_dict


def raw_reference_timeseries(reference_timeseries_array):
    mean = np.mean(reference_timeseries_array, axis = 1)
    std = np.std(reference_timeseries_array, axis = 1)
    upper = mean + std
    lower = mean - std

    result = {'upper': upper, 'mean': mean, 'lower': lower, 'std': std}

    return result

def dynamic_reference_update(current_season_NDVI, reference_full_coverage_timeseries, reference_no_coverage_timeseries):
    
    """
    Update reference full coverage and no coverage time series based on the current seasonal NDVI data.

    Parameters:
    - current_season_NDVI: 3D NumPy array containing NDVI values for the current season.
    - reference_full_coverage_timeseries: 1D NumPy array representing the reference full coverage time series.
    - reference_no_coverage_timeseries: 1D NumPy array representing the reference no coverage time series.

    Returns:
    - updated_ref_full_coverage: Updated reference full coverage time series.
    - updated_ref_no_coverage: Updated reference no coverage time series.
    """


    # acquire current seasonal NDVI array
    seasonal_NDVI_stack = current_season_NDVI.copy()
    
    # reduce the 3D array down to 12 timestep NDVI values to pixel coordinates
    reshaped_NDVI = seasonal_NDVI_stack.reshape(seasonal_NDVI_stack.shape[0], -1) # assuming that axis 0 stores timestep information

    # select against all nan timeseries
    valid_pixel_indices = ~np.any(np.isnan(reshaped_NDVI), axis=0)
    filtered_reshaped_NDVI = reshaped_NDVI[:, valid_pixel_indices]

    # update reference_full_coverage_timeseries by taking timestep wise maximum if its higher than full coverage 
    seasonal_max_timeseries = np.max(filtered_reshaped_NDVI, axis = 1).flatten()
    updated_ref_full_coverage = np.where(reference_full_coverage_timeseries < seasonal_max_timeseries, seasonal_max_timeseries, reference_full_coverage_timeseries)    

    # update reference_no_coverage_timeseries by taking timestep wise minimum if it's lower than no coverage
    seasonal_min_timeseries = np.min(filtered_reshaped_NDVI, axis = 1).flatten()
    updated_ref_no_coverage = np.where(reference_no_coverage_timeseries > seasonal_min_timeseries, seasonal_min_timeseries, reference_no_coverage_timeseries)    

    return updated_ref_full_coverage, updated_ref_no_coverage


def area_stat(weighted_dict):
    
    """
    Calculate the mean and standard deviation of values for each date in a weighted dictionary.

    Parameters:
    weighted_dict (dict): A dictionary where keys represent dates and values are lists of numeric values.

    Returns:
    weighted_sum_mean_list (list): A list of mean values, one for each date in the dictionary.
    weighted_sum_std_list (list): A list of standard deviation values, one for each date in the dictionary.

    Example usage:
    weighted_data = {
        '2023-01-01': [10, 12, 15, 20],
        '2023-01-02': [11, 13, 14, 19]
    }
    mean_list, std_list = area_stat(weighted_data)
    """
    # Initialize variables to store meana and std
    weighted_sum_mean_list = []
    weighted_sum_std_list = []
    
    # Iterate through keys/date 
    for date in weighted_dict.keys():
        # calculate the mean 
        weighted_sum_mean_list.append(np.mean(weighted_dict[date]))
        # calculate the std
        weighted_sum_std_list.append(np.std(weighted_dict[date]))

    return weighted_sum_mean_list, weighted_sum_std_list


