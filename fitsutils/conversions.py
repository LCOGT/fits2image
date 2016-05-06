import logging
import os

from PIL import ImageFont, ImageDraw, Image

from fitsutils.scaling import get_scaled_image


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


def fits_to_jpg(path_to_fits, path_to_jpg, width=200, height=200, progressive=False, label_text='', label_font='DejaVuSansMono.ttf',
                zmin=None, zmax=None, gamma_adjust=2.5, contrast=0.1, quality=95):
    '''Create a jpg from a fits file
    '''
    if not os.path.exists(path_to_fits):
        logging.warning('fits file {} does not exist'.format(path_to_fits))
        return False

    im = get_scaled_image(path_to_fits, zmin=zmin, zmax=zmax, contrast=contrast, gamma_adjust=gamma_adjust, flip_v=True)
    im.thumbnail((width, height), Image.ANTIALIAS)
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
    im.thumbnail((width, height), Image.ANTIALIAS)
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
