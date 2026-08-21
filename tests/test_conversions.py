'''End-to-end conversion: the Pillow 10 label regression, and orientation through the API.'''
import os
import shutil
import tempfile
import unittest

from PIL import Image, ImageFont

from fits2image.conversions import fits_to_img, fits_to_jpg, fits_to_tif
from tests.helpers import NORTH, brightest_pixel, lco_cd, write_fits

LABEL_FONT = 'DejaVuSansMono.ttf'


def font_is_available():
    try:
        ImageFont.truetype(LABEL_FONT, 20)
        return True
    except OSError:
        return False


class ConversionTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def frame(self, name='frame.fits', rotation=0.0, naxis=256):
        return write_fits(self.path(name), lco_cd(rotation, True, False), naxis=naxis)


@unittest.skipUnless(font_is_available(), '{} not installed'.format(LABEL_FONT))
class TestLabelling(ConversionTestCase):
    '''Pillow 10 removed FreeTypeFont.getsize, which fits_to_img called through
    _add_label. Only IOError was caught, so the AttributeError failed the whole
    conversion - and the site passes annotate=true for its main 900px quick look,
    while the central thumbnail service never passes a label at all.'''

    def test_a_labelled_jpeg_is_produced(self):
        out = self.path('labelled.jpg')

        self.assertTrue(fits_to_jpg(self.frame(), out, width=128, height=128,
                                    label_text='cpt1m010-fa14-20260818-0296-e01'))

        self.assertGreater(os.path.getsize(out), 0)

    def test_the_label_is_actually_drawn(self):
        plain, labelled = self.path('plain.jpg'), self.path('labelled.jpg')
        frame = self.frame()

        fits_to_jpg(frame, plain, width=128, height=128)
        fits_to_jpg(frame, labelled, width=128, height=128, label_text='cpt1m010-fa14')

        self.assertNotEqual(Image.open(plain).tobytes(), Image.open(labelled).tobytes())

    def test_an_over_wide_label_shrinks_instead_of_looping_forever(self):
        out = self.path('tiny.jpg')

        self.assertTrue(fits_to_jpg(self.frame(), out, width=16, height=16,
                                    label_text='a' * 400))

    def test_a_missing_font_does_not_fail_the_conversion(self):
        out = self.path('nofont.jpg')

        self.assertTrue(fits_to_jpg(self.frame(), out, width=128, height=128,
                                    label_text='label', label_font='NoSuchFont.ttf'))

        self.assertGreater(os.path.getsize(out), 0)


class TestOrientation(ConversionTestCase):

    def north_position(self, out):
        image = Image.open(out)
        _, y = brightest_pixel(image)
        return y, image.size[1]

    def test_wcs_puts_north_in_the_top_half_of_a_rotation_180_frame(self):
        '''fa14 and the rest of the 180 degree Sinistro group.'''
        out = self.path('wcs.jpg')

        fits_to_jpg(self.frame(rotation=180.0), out, width=128, height=128, orient='wcs')

        y, height = self.north_position(out)
        self.assertLess(y, height / 2, 'north did not end up in the top half')

    def test_legacy_leaves_north_at_the_bottom_of_the_same_frame(self):
        out = self.path('legacy.jpg')

        fits_to_jpg(self.frame(rotation=180.0), out, width=128, height=128, orient='legacy')

        y, height = self.north_position(out)
        self.assertGreater(y, height / 2, 'legacy orientation unexpectedly changed')

    def test_legacy_is_the_default(self):
        frame = self.frame(rotation=180.0)
        default, legacy = self.path('default.jpg'), self.path('legacy.jpg')

        fits_to_jpg(frame, default, width=128, height=128)
        fits_to_jpg(frame, legacy, width=128, height=128, orient='legacy')

        self.assertEqual(Image.open(default).tobytes(), Image.open(legacy).tobytes())

    def test_orient_reaches_the_tiff_path_too(self):
        frame = self.frame(rotation=180.0)
        legacy, wcs = self.path('legacy.tif'), self.path('wcs.tif')

        fits_to_tif(frame, legacy, width=128, height=128, orient='legacy')
        fits_to_tif(frame, wcs, width=128, height=128, orient='wcs')

        self.assertNotEqual(Image.open(legacy).tobytes(), Image.open(wcs).tobytes())


class TestUnchangedBehaviour(ConversionTestCase):
    '''The paths this story is not meant to touch.'''

    def test_a_missing_file_returns_false(self):
        self.assertFalse(fits_to_jpg(self.path('nope.fits'), self.path('out.jpg')))

    def test_a_colour_stack_still_works(self):
        frames = [self.frame('r.fits'), self.frame('g.fits'), self.frame('b.fits')]
        out = self.path('colour.jpg')

        self.assertTrue(fits_to_img(frames, out, 'jpeg', width=64, height=64, color=True))

        self.assertEqual(Image.open(out).mode, 'RGB')

    def test_a_colour_stack_needs_exactly_three_frames(self):
        frames = [self.frame('r.fits'), self.frame('g.fits')]

        self.assertFalse(fits_to_img(frames, self.path('out.jpg'), 'jpeg', color=True))

    def test_mismatched_zmin_list_is_rejected(self):
        frames = [self.frame('r.fits'), self.frame('g.fits'), self.frame('b.fits')]

        self.assertFalse(fits_to_img(frames, self.path('out.jpg'), 'jpeg',
                                     color=True, zmin=[1, 2], zmax=[3, 4, 5]))


if __name__ == '__main__':
    unittest.main()
