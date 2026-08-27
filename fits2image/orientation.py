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

# Tells orient_image to work the transform out from the frame's own header, which a
# caller that has already resolved one across a group of frames does not want.
DERIVE_FROM_HEADER = object()


def get_cd_matrix(header):
    ''' Read the CD matrix out of a FITS header.
    :param header: an astropy Header, or any mapping supporting subscript access
    :return: a 2x2 numpy array, or None if the header carries no usable WCS
    '''
    if header is None:
        return None
    try:
        cd11, cd12, cd21, cd22 = (float(header[keyword]) for keyword in CD_KEYWORDS)
    except (KeyError, TypeError, ValueError):
        return None
    cd = np.array([[cd11, cd12],
                   [cd21, cd22]])
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


def orient_image(image, header, flip_v=True, frame='', ops=DERIVE_FROM_HEADER):
    ''' Put north up and east left, falling back to a fixed vertical flip.
    :param image: Pillow Image, as produced by Image.fromarray
    :param header: FITS header of the frame
    :param flip_v: the fallback flip, applied when there is no usable WCS
    :param frame: names the frame in the fallback warning, which is otherwise unactionable
                  for a service converting thousands of them
    :param ops: the transform to apply, for a caller that has already resolved one across a
                group of frames. Defaults to deriving it from this frame's own header.
    :return: a new Pillow Image
    '''
    if ops is DERIVE_FROM_HEADER:
        ops = orientation_ops(header)

    if ops is not None:
        return apply_orientation(image, ops)

    logging.warning('No usable WCS in %s, falling back to flip_v=%s', frame or 'header', flip_v)
    if flip_v:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image
