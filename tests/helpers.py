'''Shared fixtures for the fits2image tests.'''
import numpy as np
from astropy.io import fits

from fits2image.orientation import CD_KEYWORDS

# Roughly the plate scale of an LCO 0m4, in degrees per pixel. The absolute value
# does not matter to any test here, only the signs and ratios inside the CD matrix.
SCALE = 2.0653e-4

NORTH = (0.0, 1.0)
EAST = (1.0, 0.0)


def lco_cd(rotation_deg=0.0, flipx=True, flipy=False, sky_pa_deg=0.0, scale=SCALE):
    '''Build a CD matrix the way the site software does.

    Mirrors WorldCoordinateSystem.CDMatrix.configureMatrix() in site-software's
    instrument-agent, so the tests exercise the real range of LCO instrument geometry
    rather than a textbook WCS.
    '''
    angle = np.radians(sky_pa_deg - rotation_deg)
    flip = -1.0 if ((not flipx) ^ flipy) else 1.0
    cpa, spa = scale * np.cos(angle), scale * np.sin(angle)
    return {'CD1_1': -flip * cpa, 'CD1_2': flip * spa,
            'CD2_1': spa, 'CD2_2': cpa}


def header_with_cd(cd, naxis=201, **extra):
    header = fits.Header()
    header['NAXIS1'] = naxis
    header['NAXIS2'] = naxis
    header.update(cd)
    header.update(extra)
    return header


def sky_offset_pixels(cd, direction, distance=60.0):
    '''Where a source `distance` pixels toward `direction` on the sky lands in the array.

    A frame with no CD matrix has no sky direction, so the blob just goes at a fixed
    offset - such frames only exist in these tests to exercise the fallback path.
    '''
    if not all(keyword in cd for keyword in CD_KEYWORDS):
        return np.array(direction) * distance
    matrix = np.array([[cd['CD1_1'], cd['CD1_2']], [cd['CD2_1'], cd['CD2_2']]])
    offset = np.linalg.inv(matrix) @ np.array(direction)
    return offset / np.linalg.norm(offset) * distance


def make_array(cd, naxis=201, blobs=((NORTH, 255),), background=0, radius=5):
    '''An array with a bright blob at each requested sky direction.'''
    array = np.full((naxis, naxis), background, dtype=np.uint16)
    centre = naxis // 2
    for direction, value in blobs:
        di, dj = sky_offset_pixels(cd, direction)
        i, j = int(round(centre + di)), int(round(centre + dj))
        array[j - radius:j + radius + 1, i - radius:i + radius + 1] = value
    return array


def write_fits(path, cd, naxis=201, blobs=((NORTH, 60000),), noise=True, **extra):
    '''A single-HDU frame, as produced at site in the flash daydir.'''
    array = make_array(cd, naxis=naxis, blobs=blobs, radius=8).astype(np.float32)
    if noise:
        rng = np.random.default_rng(20260819)
        array += rng.normal(1000.0, 40.0, array.shape)
    header = fits.Header()
    header.update(cd)
    header.update(extra)
    fits.writeto(path, array, header, overwrite=True)
    return path


def write_multi_hdu_fits(path, cd, naxis=64, **extra):
    '''A Sinistro-shaped frame: full header on an empty primary, pixels in extensions.

    The primary carries the WCS and SATURATE; the quadrant extensions carry neither.
    '''
    rng = np.random.default_rng(20260819)
    primary_header = fits.Header()
    primary_header.update(cd)
    primary_header['SATURATE'] = 65535
    primary_header.update(extra)
    primary = fits.PrimaryHDU(data=np.zeros((0, 0), dtype=np.int16), header=primary_header)
    quadrants = [fits.ImageHDU(data=rng.normal(1000.0, 40.0, (naxis, naxis)).astype(np.float32))
                 for _ in range(4)]
    fits.HDUList([primary] + quadrants).writeto(path, overwrite=True)
    return path


def brightest_pixel(image):
    '''(x, y) of the brightest pixel, in screen coordinates - y grows DOWNWARD.'''
    array = np.asarray(image.convert('L'))
    j, i = np.unravel_index(np.argmax(array), array.shape)
    return int(i), int(j)
