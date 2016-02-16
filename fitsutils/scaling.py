import math

import fitsio
import numpy as np
from PIL import Image


def get_scaled_image(path_to_fits, zmin=None, zmax=None, contrast=0.1, gamma_adjust=2.5, flip_v=True):
    ''' Helper function to get a scaled PIL Image given a fits or compressed fits file path and scale parameters
    :param path_to_fits:
    :param zmin:
    :param zmax:
    :param contrast:
    :param gamma_adjust:
    :param flip_v: Should the image be flipped vertically?
    :return:
    '''
    if zmin or zmax:
        data, header = get_reduced_dimensionality_data(path_to_fits)
        scaled_data = linear_scale(data, zmin, zmax, gamma_adjust=gamma_adjust)
    else:
        scaled_data = auto_scale(path_to_fits, contrast=contrast, gamma_adjust=gamma_adjust)
    im = Image.fromarray(scaled_data)
    if flip_v:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

    return im


def simple_line_fit(sample_data):
    nsamples = len(sample_data)

    xstart = int(0.3 * nsamples)
    xend = int(0.7 * nsamples)
    x_range = xend - xstart

    ystart = sample_data[xstart]
    yend = sample_data[xend]
    y_range = yend - ystart

    slope = float(y_range) / float(x_range)
    y_intercept= sample_data[nsamples/2]

    return slope, y_intercept


def least_squares_line_fit(sample_data, max_iterations=5, min_fit=0.5):
    nsamples = len(sample_data)
    max_masked_samples = nsamples * (1 - min_fit)

    # Contains True if sample is included in fit False is sample is excluded
    sample_mask = np.ones(len(sample_data), dtype=bool)
    fit_samples = sample_data[sample_mask]
    nmasked = 0

    last_fit = None
    last_fit_rms = None
    for i in range(max_iterations):
        nmasked += len(np.where(sample_mask == False)[0])
        # Use min and max to of samples to determine slope and y_intercept
        if nmasked >= max_masked_samples:
            slope = (sample_data[-1] - sample_data[0]) / nsamples
            y_intercept = sample_data[-1] - slope * nsamples
            fit_samples = sample_data
            break

        fit_samples = fit_samples[sample_mask]
        sample_mask = np.ones(len(fit_samples), dtype=bool)
        nfitsamples = len(fit_samples)
        x = np.array(range(nfitsamples))
        y = np.array(fit_samples)

        A = np.vstack([x, np.ones(nfitsamples)]).T
        result = np.linalg.lstsq(A,y)
        slope, y_intercept = result[0]
        residuals = result[1]

        # Need to check residual and remove outliers before refit if residual is large
        mean_residual = residuals / nfitsamples
        rms = math.sqrt(mean_residual)
        fitline = np.array(range(nfitsamples))
        fitline = fitline * slope
        fitline = fitline + y_intercept
        over_threshold = fitline + rms
        under_threshold = fitline - rms

        sample_mask[np.where(y > over_threshold)[0]] = False
        sample_mask[np.where(y < under_threshold)[0]] = False

        if last_fit is None or rms < last_fit_rms:
            last_fit = (slope, y_intercept, i+1, nfitsamples, rms, fit_samples)
            last_fit_rms = rms
        else:
            break


    return last_fit


def extract_samples(data, header, nsamples=2000):
    ''' Extract a set of samples from a fits image and
    return a sorted numpy array of results
    '''
    flat_data = data.flatten()

    sample_stride = (header.get('NAXIS1') * header.get('NAXIS2')) / nsamples
    samples = flat_data[sample_stride::sample_stride]
    samples.sort()

    return samples


def calc_zscale_min_max(samples, contrast= 0.1, iterations=5):
    nsamples = len(samples)
    slope, y_intecept, used_iterations, nfitsamples, rms, fit_samples = least_squares_line_fit(samples, max_iterations=iterations)
    print slope, y_intecept, used_iterations, nfitsamples, rms

    zmin = samples[0]
    zmax = samples[nsamples - 1]
    if contrast > 0.0:
        slope = slope / contrast

    fitted_dy = slope * nsamples/2
    zmin = max(zmin, y_intecept - fitted_dy)
    zmax = min(zmax, y_intecept + fitted_dy)

    return zmin, zmax, rms


def linear_scale(data, zmin, zmax, max_val=255, gamma_adjust=2.5):
    '''Apply a linear rescale of the supplied fits images cliping between
    zmin and zmax and writing to the supplied outfile.
    '''
    scale = float(max_val) / (float(zmax) - float(zmin))
    adjust = scale * zmin

    data = data.astype('float')
    data.clip(zmin, zmax, data)
    data *= scale
    data -= adjust
    data.round(out=data)
    gamma_lookup_table = gamma_adjust_table(data.dtype, max_val=max_val, gamma_adjust=gamma_adjust)

    #data = gamma_lookup_table[data] the below line is supposedly faster
    np.take(gamma_lookup_table, data.astype('int64'), out=data)

    data = data.astype('uint8')
    return data


def gamma_adjust_table(dtype, max_val=255.0, min_val=0.0, gamma_adjust=2.5):
    '''Creates a lookup table for a gamma adjustment
    '''
    size = int(max_val) - int(min_val) + 1
    gamma_lookup_table = xrange(size)
    gamma_lookup_table = [int(size * (math.pow(float(i)/float(size), 1.0/float(gamma_adjust)))) for i in gamma_lookup_table]

    return np.array(gamma_lookup_table, dtype=dtype)


def auto_scale(path_to_frame, nsamples=2000, max_val=255, contrast=0.1, gamma_adjust=2.5, max_fit_iterations=1):
    '''Uses a zscale fit and rescales the image accordingly by calling
    linear scale
    '''
    data, header = get_reduced_dimensionality_data(path_to_frame)
    samples = extract_samples(data, header, nsamples)
    median = np.median(samples)
    zmin, zmax, rms = calc_zscale_min_max(samples, contrast=contrast, iterations=max_fit_iterations)
    print {'median':median, 'zmin':zmin, 'zmax':zmax, 'fit_rms':rms}
    return linear_scale(data, median, zmax, max_val, gamma_adjust)


def get_reduced_dimensionality_data(path_to_frame):
    '''
    Reduce the dimensionality of the data by 1. For sinistro images, this will give the first quadrant
    :param path_to_frame: path to fits file
    :return: header and modified data from fitsio
    '''
    data, header = fitsio.read(path_to_frame, header=True, mode='r')
    while len(data.shape) > 2:
        data = data[0]
    return data, header


def percentile_scale(path_to_frame, lower_percentile=5.0, upper_percentile=99.0):
    ''' This is not currently used, but can be compared against the auto_scale /zscaled method of image scaling
        It doesn't appear to work as well with nebulas/galaxies as the zscale method but works well for stars
    :param path_to_frame:
    :param lower_percentile:
    :param upper_percentile:
    :return:
    '''
    data = fitsio.read(path_to_frame, mode='r')
    (lower_threshold, upper_threshold) = np.percentile(data, [lower_percentile, upper_percentile])

    data = data.astype('float')
    data = ((data-lower_threshold)/(upper_threshold-lower_threshold))*255.0
    data[data < 0] = 0
    data[data > 255] = 255
    data = data.astype('uint8')
    return data