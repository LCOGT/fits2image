'''
    Derive a display orientation from a frame's WCS, so that images taken with
    different instruments, rotations and focal-plane flips can be compared by eye.
    The transform is snapped to the nearest 90 degrees.

    Conventions, stated because every bug in this area is a convention bug:

    * array index (i, j): i is FITS axis 1 - 1 (column), j is FITS axis 2 - 1 (row)
    * Image.fromarray puts array row j=0 at the TOP, so on screen right is +i and up is -j
    * CD maps (di, dj) to (dxi, deta), with xi toward east (increasing RA) and eta toward north
'''
import logging

import numpy as np
from PIL import Image


CD_KEYWORDS = ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')

# Below this the CD matrix is not invertible in any useful sense.
MIN_DETERMINANT = 1e-20


def get_cd_matrix(header):
    ''' Read the CD matrix out of a FITS header.
    :param header: an astropy Header, or any mapping supporting .get()
    :return: a 2x2 numpy array, or None if the header carries no usable WCS
    '''
    if header is None:
        return None
    try:
        cd = np.array([[float(header['CD1_1']), float(header['CD1_2'])],
                       [float(header['CD2_1']), float(header['CD2_2'])]])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(cd)):
        return None
    if abs(np.linalg.det(cd)) < MIN_DETERMINANT:
        return None
    return cd


def orientation_ops(header):
    ''' Work out the transform that puts north up and east left, to the nearest 90 degrees.
    :param header: FITS header of the frame
    :return: (mirror, k) - mirror the image left-right if mirror is True, THEN apply k
             counter-clockwise quarter turns. None if there is no usable WCS, so that
             callers can fall back to their previous behaviour.
    '''
    cd = get_cd_matrix(header)
    if cd is None:
        return None

    inverse = np.linalg.inv(cd)
    north = inverse @ np.array([0.0, 1.0])
    east = inverse @ np.array([1.0, 0.0])

    # Into the screen frame of Image.fromarray, where up is -j.
    n = np.array([north[0], -north[1]])
    e = np.array([east[0], -east[1]])

    # Standard astronomical parity is north up / east left, for which the cross
    # product z of (north x east) is positive. Anything else needs one mirror.
    mirror = bool((n[0] * e[1] - n[1] * e[0]) < 0)

    # Mirror first - it does not commute with rotation, so the angle below has to
    # be measured on the already-corrected parity.
    if mirror:
        n = np.array([-n[0], n[1]])

    # Position angle of north, counter-clockwise from screen-up. Image.ROTATE_90
    # turns the content counter-clockwise, moving a feature at angle theta to
    # theta + 90, so bringing north to zero needs 90k = -theta.
    theta = np.degrees(np.arctan2(-n[0], n[1]))
    k = int(np.rint(-theta / 90.0)) % 4

    return mirror, k


def apply_orientation(image, ops):
    ''' Apply the transform returned by orientation_ops to a Pillow image.
    :param image: Pillow Image
    :param ops: (mirror, k) as returned by orientation_ops
    :return: a new Pillow Image. Note that an odd k swaps width and height.
    '''
    mirror, k = ops
    if mirror:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    for _ in range(k):
        image = image.transpose(Image.ROTATE_90)
    return image


def orient_image(image, header, orient='legacy', flip_v=True):
    ''' Orient an image either from its WCS or by the legacy fixed vertical flip.
    :param image: Pillow Image, as produced by Image.fromarray
    :param header: FITS header of the frame
    :param orient: 'wcs' to put north up from the CD matrix, 'legacy' for the fixed flip
    :param flip_v: the legacy vertical flip, also used as the fallback when orient='wcs'
                   but the frame has no usable WCS
    :return: a new Pillow Image
    '''
    if orient not in ('wcs', 'legacy'):
        raise ValueError("orient must be 'wcs' or 'legacy', not {!r}".format(orient))

    if orient == 'wcs':
        ops = orientation_ops(header)
        if ops is not None:
            return apply_orientation(image, ops)
        logging.warning('No usable WCS in header, falling back to flip_v=%s', flip_v)

    if flip_v:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image
