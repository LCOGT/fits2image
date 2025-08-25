from fits2image.scaling import get_scaled_image, stack_images, quick_scale_image, DEFAULT_GAMMA_LUT

import logging
import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np


SCALE_ALGORITHMS = {
    'zscale': ['zmin', 'zmax']
}
BLENDING_ALGORITHMS = ['sum', 'min', 'max', 'multiply', 'screen', 'overlay']
FILE_TYPES = ['jpeg', 'tiff']


def _add_label(image, label_text, label_font):
    '''
        Writes label_text into Pillow Image image.
    :param image: Pillow Image Object - will be modified with the label_text
    :param label_text:
    :param label_font: The true type font on the system to use in the label. throws IOError if it can't be found.
    :return:
    '''
    (width, height) = image.size
    font_size = 20
    offset = 4
    font = ImageFont.truetype(label_font, font_size)
    (font_width, font_height) = font.getsize(label_text)
    while font_width > (int(width)-offset):
        font_size -= 1
        font = ImageFont.truetype(label_font, font_size)
        (font_width, font_height) = font.getsize(label_text)

    d = ImageDraw.Draw(image)
    d.text((offset, int(height) - offset - font_height), label_text, font=font, fill=255)


def fits_to_img(path_to_fits, path_to_output, file_type, width=200, height=200, progressive=False, label_text='', label_font='DejaVuSansMono.ttf',
                zmin=None, zmax=None, gamma_adjust=2.5, contrast=0.1, quality=95, color=False, percentile=99.5, median=False):
    '''
        Create a img of file_type from a fits file
        :param path_to_fits a single file or list (if color=True)
        :param path_to_output: the path to save the output image
        :param file_type: the type of image to save (e.g. 'jpeg', 'TIFF')
        :param width: the width of the output image
        :param height: the height of the output image
        :param progressive: should the image be saved as progressive?
        :param label_text: text to add to the image
        :param label_font: the font to use for the label
        :param zmin: the minimum value to scale the image to - should be a list of the same length as path_to_fits if path_to_fits is a list
        :param zmax: the maximum value to scale the image to - should be a list of the same length as path_to_fits if path_to_fits is a list
        :param gamma_adjust: the gamma adjustment to apply to the image
        :param contrast: the contrast to apply to the image
        :param quality: the quality of the output image
        :param color: should the output image be color?
        :param percentile: the percentile to use for the median calculation
        :param median: should the median be recalculated?
    '''

    # If path_to_fits is not a list, make it a list so that we can loop through it
    if type(path_to_fits) != list:
        path_to_fits = [path_to_fits]

    # Make sure zmin and zmax match the length of path_to_fits if they haven't been specified,
    # since we loop through them and pass them to the scale function
    if zmin is None:
        zmin = [None] * len(path_to_fits)
    if zmax is None:
        zmax = [None] * len(path_to_fits)

    # If zmin and zmax are not lists, make them lists so that we can handle the case where we only have one image
    else:
        if type(zmin) != list:
            zmin = [zmin]
        if type(zmax) != list:
            zmax = [zmax]

    # Finally, check that zmin and zmax are set correctly to the length of the path_to_fits
    if len(zmin) != len(path_to_fits):
        logging.error('zmin must be the same length as path_to_fits')
        return False
    if len(zmax) != len(path_to_fits):
        logging.error('zmax must be the same length as path_to_fits')
        return False

    scaled_images = []

    for path, zmin_entry, zmax_entry in zip(path_to_fits, zmin, zmax):
        try:
            scaled_images.append(
                get_scaled_image(path, zmin=zmin_entry, zmax=zmax_entry, contrast=contrast, gamma_adjust=gamma_adjust, flip_v=True, percentile=percentile, median=median)
            )
        except FileNotFoundError:
            logging.error('File {} not found'.format(path))
            return False

    if color:
        if len(scaled_images) != 3:
            logging.error(f'Need exactly 3 FITS files (RVB) to create a color {file_type}')
            return False
        scaled_images = [stack_images(scaled_images)]

    for idx, im in enumerate(scaled_images):
        im.thumbnail((width, height), Image.LANCZOS)
        if label_text:
            try:
                _add_label(im, label_text, label_font)
            except IOError:
                # just log a warning and continue - its okay if you cant write a label
                logging.warning('font {} could not be found on the system. Ignoring label text.'.format(label_font))

        try:
            path_only = os.path.dirname(path_to_output)
            filename = path_to_output
            if not os.path.exists(path_only):
                os.makedirs(path_only)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            if idx > 0:
                filename = '{0}-{1}'.format(filename, idx)
            im.save(filename, file_type, quality=quality, progressive=progressive)
        except IOError as ioerr:
            logging.warning(f'Error saving {file_type}: {path_to_output}. Reason: {str(ioerr)}')
            return False
    return True


def fits_to_zoom_slice_jpg(path_to_fits, path_to_jpg, row=0, col=0, side=200, zlevel=0, zfactor=1.25, progressive=False,
                           label_text='', label_font='DejaVuSansMono.ttf', zmin=None, zmax=None, gamma_adjust=2.5,
                           contrast=0.1, quality=75):
    '''Create a slice of a zoomed in jpg from a fits file
    '''
    if not os.path.exists(path_to_fits):
        logging.warning('fits file {} does not exist'.format(path_to_fits))
        return False

    im = get_scaled_image(path_to_fits, zmin=zmin, zmax=zmax, contrast=contrast, gamma_adjust=gamma_adjust, flip_v=True)
    height = side
    width = side
    # zoom scale is display px per image px
    zoom_scale = zfactor**zlevel
    # step is in image px
    step = int(side / zoom_scale)
    x = int(col * step)
    y = int(row * step)
    im = im.crop((x, y, x+step, y+step))
    im.thumbnail((width, height), Image.LANCZOS)
    if label_text:
        try:
            _add_label(im, label_text, label_font)
        except IOError:
            # just log a warning and continue - its okay if you cant write a label
            logging.warning('font {} could not be found on the system. Ignoring label text.'.format(label_font))

    try:
        path_only = os.path.dirname(path_to_jpg)
        if not os.path.exists(path_only):
            os.makedirs(path_only)
        im.save(path_to_jpg, 'jpeg', quality=quality, progressive=progressive)
    except IOError as ioerr:
        logging.warning('Error saving jpeg: {}. Reason: {}'.format(path_to_jpg, str(ioerr)))
        return False
    return True

def fits_to_tif(path_to_fits, path_to_tif, width=200, height=200, contrast=0.1, gamma_adjust=2.5, quality=100, percentile=99.5, median=False, progressive=False):
    '''
        Create a tif from a fits file
    '''
    return fits_to_img(path_to_fits, path_to_tif, 'TIFF', width=width, height=height, contrast=contrast, gamma_adjust=gamma_adjust, quality=quality, percentile=percentile, median=median, progressive=progressive)

def fits_to_jpg(path_to_fits, path_to_jpg, width=200, height=200, progressive=False, label_text='', label_font='DejaVuSansMono.ttf',
                zmin=None, zmax=None, gamma_adjust=2.5, contrast=0.1, quality=95, color=False, percentile=99.5, median=False):
    '''
        Create a jpg from a fits file
    '''
    return fits_to_img(path_to_fits, path_to_jpg, 'jpeg', width=width, height=height, progressive=progressive, label_text=label_text, label_font=label_font, zmin=zmin, zmax=zmax, gamma_adjust=gamma_adjust, contrast=contrast, quality=quality, color=color, percentile=percentile, median=median)

def multi_fits_to_img(input_fits, path_to_output, blending_algorithm='sum', width=200, height=200, file_type='jpeg', progressive=False, quality=95):
    '''
    Create a jpg from a list of dicts of input parameters. Each input should look like this:
    {
        'fits_path': path-to-fits (either this or path),
        'fits_data': raw data as numpy.ndarray (either this or path),
        'scale_algorithm': 'zscale',
        'color': (r, g, b),
        'zmin': zmin (only for zscale),
        'zmax': zmax (only for zscale)
    }
    file_type: 'jpeg' or 'tiff'
    blending_algorithm: The algorithm used to combine the color channels from input images into the output rgb
        options - sum, max, min, screen, multiply, overlay.
    '''
    # Check that all inputs are present or raise and exception
    for input_dict in input_fits:
        if 'fits_path' not in input_dict and 'fits_data' not in 'input_dict':
            raise ValueError("Each input must have either 'fits_path' or 'fits_data' set")
        if 'color' not in input_dict or len(input_dict['color']) != 3 or any(color > 1 or color < 0 for color in input_dict['color']):
            raise ValueError("Each input must have a 'color' field with 3 elements [r,g,b], where each element is between 0 and 1")
        if 'scale_algorithm' not in input_dict or input_dict['scale_algorithm'] not in SCALE_ALGORITHMS.keys():
            raise ValueError(f"Each input must have a 'scale_algorithm' field set, one of {', '.join(SCALE_ALGORITHMS.keys())}")
        if any(field not in input_dict for field in SCALE_ALGORITHMS[input_dict['scale_algorithm']]):
            raise ValueError(f"The scale_algorithm {input_dict['scale_algorithm']} requires additional fields {', '.join(SCALE_ALGORITHMS[input_dict['scale_algorithm']])} to be set")
        if blending_algorithm not in BLENDING_ALGORITHMS:
            raise ValueError(f"Invalid 'blending_algorithm' {blending_algorithm}. Must be one of: {', '.join(BLENDING_ALGORITHMS)}")
        if file_type not in FILE_TYPES:
            raise ValueError(f"Invalid 'file_type' {file_type}. Must be one of: {', '.join(FILE_TYPES)}")

    largest_dtype = np.dtype('uint8')
    for input_dict in input_fits:
        # First get the scaled version of the image given your scaling algorithm and params
        input_dict['scaled_image'] = quick_scale_image(input_dict)
        largest_dtype = max(largest_dtype, input_dict['scaled_image'].dtype)

    # Then check if images are same size and if not crop them to be the min_width/min_height
    shapes = [input_dict['scaled_image'].shape for input_dict in input_fits]
    widths = [shape[0] for shape in shapes]
    heights = [shape[1] for shape in shapes]
    min_width = min(widths)
    min_height = min(heights)
    if len(set(shapes)) != 1:
        # Otherwise we must crop them to get them to be the same shape
        for input_dict in input_fits:
            shape = input_dict['scaled_image'].shape
            left = (shape[0] - min_width) // 2
            top = (shape[1] - min_height) // 2
            right = left + min_width
            bottom = top + min_height
            input_dict['scaled_image'] = input_dict['scaled_image'][left:right, top:bottom]

    # Setup the output array based on the blending_algorithm
    match blending_algorithm:
        case 'multiply':
            # Multiply multiplies, so should start as ones, not zeros
            combined_image = np.ones((min_width, min_height, 3), dtype=largest_dtype)
        case 'screen':
            # Screen multiplies, so should start as ones, not zeros
            combined_image = np.ones((min_width, min_height, 3), dtype=largest_dtype)
        case 'overlay':
            # Overlay should start with the first layer as the base layer and go from there
            combined_image = np.ndarray((min_width, min_height, 3), dtype=largest_dtype)
            input_dict = input_fits[0]
            color = input_dict['color']
            input_dict['scaled_image'] /= 255
            combined_image[:, :, 0] = input_dict['scaled_image'] * color[0]
            combined_image[:, :, 1] = input_dict['scaled_image'] * color[1]
            combined_image[:, :, 2] = input_dict['scaled_image'] * color[2]
        case 'min':
            # Start with the first image and take the minimum from there
            combined_image = np.ndarray((min_width, min_height, 3), dtype=largest_dtype)
            input_dict = input_fits[0]
            color = input_dict['color']
            combined_image[:, :, 0] = input_dict['scaled_image'] * color[0]
            combined_image[:, :, 1] = input_dict['scaled_image'] * color[1]
            combined_image[:, :, 2] = input_dict['scaled_image'] * color[2]
        case _:
            # All others can start with an empty array since they are additive
            combined_image = np.ndarray((min_width, min_height, 3), dtype=largest_dtype)

    # Then combine all the scaled images in r, g, b, channels of a final image stack
    match blending_algorithm:
            case 'sum':
                # Sum the pixel contributions
                for input_dict in input_fits:
                    color = input_dict['color']
                    if color[0] > 0:
                        combined_image[:,:,0] += input_dict['scaled_image'] * color[0]
                    if color[1] > 0:
                        combined_image[:,:,1] += input_dict['scaled_image'] * color[1]
                    if color[2] > 0:
                        combined_image[:,:,2] += input_dict['scaled_image'] * color[2]
                # Clip the data back to 0 to 255 at the end since it could go over
                combined_image.clip(0, 255, combined_image)
            case 'min':
                # Take the minimum value for each element
                for index, input_dict in enumerate(input_fits):
                    color = input_dict['color']
                    if index > 0:
                        if color[0] > 0:
                            combined_image[:,:,0] = np.minimum(combined_image[:,:,0], input_dict['scaled_image'] * color[0], out=combined_image[:,:,0])
                        if color[1] > 0:
                            combined_image[:,:,1] = np.minimum(combined_image[:,:,1], input_dict['scaled_image'] * color[1], out=combined_image[:,:,1])
                        if color[2] > 0:
                            combined_image[:,:,2] = np.minimum(combined_image[:,:,2], input_dict['scaled_image'] * color[2], out=combined_image[:,:,2])
            case 'max':
                # Take the maximum value for each element
                for input_dict in input_fits:
                    color = input_dict['color']
                    if color[0] > 0:
                        combined_image[:,:,0] = np.maximum(combined_image[:,:,0], input_dict['scaled_image'] * color[0], out=combined_image[:,:,0])
                    if color[1] > 0:
                        combined_image[:,:,1] = np.maximum(combined_image[:,:,1], input_dict['scaled_image'] * color[1], out=combined_image[:,:,1])
                    if color[2] > 0:
                        combined_image[:,:,2] = np.maximum(combined_image[:,:,2], input_dict['scaled_image'] * color[2], out=combined_image[:,:,2])
            case 'multiply':
                for input_dict in input_fits:
                    color = input_dict['color']
                    input_dict['scaled_image'] /= 255
                    combined_image[:, :, 0] *= (input_dict['scaled_image'] * color[0])
                    combined_image[:, :, 1] *= (input_dict['scaled_image'] * color[1])
                    combined_image[:, :, 2] *= (input_dict['scaled_image'] * color[2])
            case 'screen':
                for input_dict in input_fits:
                    color = input_dict['color']
                    input_dict['scaled_image'] /= 255
                    combined_image[:, :, 0] *= (1.0 - (input_dict['scaled_image'] * color[0]))
                    combined_image[:, :, 1] *= (1.0 - (input_dict['scaled_image'] * color[1]))
                    combined_image[:, :, 2] *= (1.0 - (input_dict['scaled_image'] * color[2]))
            case 'overlay':
                for index, input_dict in enumerate(input_fits):
                    # Skip first index since that is the base layer for overlay blending
                    if index > 0:
                        input_dict['scaled_image'] /= 255
                        color = input_dict['color']
                        combined_image[:, :, 0] = np.where(combined_image[:, :, 0] > 0.5, 1 - (2 * (1 - combined_image[:, :, 0]) * (1.0 - (input_dict['scaled_image'] * color[0]))), 2 * combined_image[:, :, 0] * ((input_dict['scaled_image'] * color[0])))
                        combined_image[:, :, 1] = np.where(combined_image[:, :, 1] > 0.5, 1 - (2 * (1 - combined_image[:, :, 1]) * (1.0 - (input_dict['scaled_image'] * color[1]))), 2 * combined_image[:, :, 1] * ((input_dict['scaled_image'] * color[1])))
                        combined_image[:, :, 2] = np.where(combined_image[:, :, 2] > 0.5, 1 - (2 * (1 - combined_image[:, :, 2]) * (1.0 - (input_dict['scaled_image'] * color[2]))), 2 * combined_image[:, :, 2] * ((input_dict['scaled_image'] * color[2])))

    # Any final manipulations to the combined image based on the
    match blending_algorithm:
        case 'multiply':
            # Multiplying occured in 0-1 space, convert back to 0-255
            combined_image *= 255
        case 'screen':
            # need to take inverse and switch back to 0 to 255
            combined_image = (1 - combined_image) * 255
        case 'overlay':
            # Overlay occured in 0-1 space, convert back to 0-255
            combined_image *= 255

    # Then apply the gamma and resize the final image to the desired size
    combined_image.round(out=combined_image)
    gamma_image = np.take(DEFAULT_GAMMA_LUT, combined_image.astype('uint8'))
    im = Image.fromarray(gamma_image)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im.thumbnail((width, height), Image.LANCZOS)
    # And save off the thumbnail
    try:
        path_only = os.path.dirname(path_to_output)
        filename = path_to_output
        if not os.path.exists(path_only):
            os.makedirs(path_only)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(filename, file_type, quality=quality, progressive=progressive)
        return True
    except IOError as ioerr:
        logging.warning(f'Error saving {file_type}: {path_to_output}. Reason: {str(ioerr)}')
        return False
