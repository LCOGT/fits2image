from fits2image.scaling import get_scaled_image, stack_images

import logging
import os
from PIL import ImageFont, ImageDraw, Image


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
