'''End-to-end conversion: the Pillow 10 label regression, and orientation through the API.'''
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageFont

from fits2image.conversions import (_add_label, _stack_orientation, fits_to_img,
                                    fits_to_jpg, fits_to_tif, fits_to_zoom_slice_jpg,
                                    multi_fits_to_img)
from fits2image.scaling import quick_scale_image
from fits2image.orientation import orientation_transform
from fits2image.scaling import get_reduced_dimensionality_data
from tests.helpers import brightest_pixel, header_with_cd, lco_cd, write_fits

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

    def assert_near(self, first, second, tolerance=2):
        '''Blob placement rounds to whole pixels, so the same sky lands within a pixel or two.'''
        self.assertLessEqual(max(abs(a - b) for a, b in zip(first, second)), tolerance,
                             '{} and {} are more than {} pixels apart'.format(first, second, tolerance))


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

    def test_a_label_with_no_ascenders_stays_inside_the_frame(self):
        '''ImageDraw.text measures y from the ascender line, so the label is placed from
        the bottom of getbbox. Placing it from bottom - top drops a string like "..."
        below the edge of the frame.'''
        image = Image.new('L', (200, 200), 0)

        _add_label(image, '...', LABEL_FONT)

        rows = np.argwhere(np.asarray(image) > 0)
        self.assertTrue(len(rows), 'the label was drawn outside the image')
        self.assertLess(rows[:, 0].max(), 200)
        self.assertGreater(rows[:, 0].max(), 190, 'the label is not against the bottom')

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

    def test_north_ends_up_in_the_top_half_of_a_rotation_180_frame(self):
        '''fa14 and the rest of the 180 degree Sinistro group.'''
        out = self.path('rotated.jpg')

        fits_to_jpg(self.frame(rotation=180.0), out, width=128, height=128)

        y, height = self.north_position(out)
        self.assertLess(y, height / 2, 'north did not end up in the top half')

    def test_instruments_at_different_rotations_agree(self):
        '''The whole point: one thumbnail orientation across the network.'''
        positions = []
        for rotation in (0.0, 90.0, 180.0, 270.0):
            out = self.path('rot{}.jpg'.format(int(rotation)))
            fits_to_jpg(self.frame('rot{}.fits'.format(int(rotation)), rotation=rotation),
                        out, width=128, height=128)
            positions.append(brightest_pixel(Image.open(out)))

        for position in positions[1:]:
            self.assert_near(positions[0], position)

    def test_a_frame_without_a_wcs_warns_and_still_converts(self):
        path = write_fits(self.path('nowcs.fits'), {}, naxis=128)
        out = self.path('nowcs.jpg')

        with self.assertLogs(level='WARNING'):
            self.assertTrue(fits_to_jpg(path, out, width=128, height=128))

        self.assertGreater(os.path.getsize(out), 0)

    def test_the_tiff_path_orients_too(self):
        out = self.path('rotated.tif')

        fits_to_tif(self.frame(rotation=180.0), out, width=128, height=128)

        y, height = self.north_position(out)
        self.assertLess(y, height / 2, 'north did not end up in the top half')


class TestZoomSlice(ConversionTestCase):

    def slice_args(self):
        return dict(row=1, col=1, side=64, zlevel=1)

    def test_a_slice_is_produced(self):
        out = self.path('slice.jpg')

        self.assertTrue(fits_to_zoom_slice_jpg(self.frame(), out, **self.slice_args()))

        self.assertGreater(os.path.getsize(out), 0)

    def test_a_missing_file_returns_false(self):
        self.assertFalse(fits_to_zoom_slice_jpg(self.path('nope.fits'), self.path('out.jpg')))

    def test_the_slice_comes_from_the_oriented_image(self):
        '''row and col index the oriented image, so the same tile is the same sky.'''
        args = dict(row=0, col=0, side=256, zlevel=0)
        upright, rotated = self.path('upright.jpg'), self.path('rotated.jpg')

        fits_to_zoom_slice_jpg(self.frame('upright.fits', rotation=0.0), upright, **args)
        fits_to_zoom_slice_jpg(self.frame('rotated.fits', rotation=180.0), rotated, **args)

        self.assert_near(brightest_pixel(Image.open(upright)),
                         brightest_pixel(Image.open(rotated)))


class TestStackOrientation(ConversionTestCase):
    '''Which transform a set of frames resolves to between them.'''

    def stack(self, *cds):
        return [write_fits(self.path('{}.fits'.format(i)), cd, naxis=256)
                for i, cd in enumerate(cds)]

    def transform_for(self, cd):
        return orientation_transform(header_with_cd(cd))

    def test_frames_that_agree_resolve_to_their_shared_transform(self):
        cd = lco_cd(180.0, True, False)

        self.assertEqual(_stack_orientation(self.stack(cd, cd, cd)), self.transform_for(cd))

    def test_a_frame_without_a_wcs_takes_the_orientation_of_its_siblings(self):
        cd = lco_cd(180.0, True, False)

        self.assertEqual(_stack_orientation(self.stack(cd, cd, {})), self.transform_for(cd))
        self.assertEqual(_stack_orientation(self.stack({}, cd, cd)), self.transform_for(cd))

    def test_a_frame_without_a_wcs_is_the_expected_case_and_does_not_warn(self):
        cd = lco_cd(180.0, True, False)

        with self.assertNoLogs(level='WARNING'):
            _stack_orientation(self.stack(cd, cd, {}))

    def test_frames_that_disagree_take_the_first_transform_and_warn(self):
        upright, turned = lco_cd(0.0, True, False), lco_cd(90.0, True, False)

        with self.assertLogs(level='WARNING'):
            transform = _stack_orientation(self.stack(upright, upright, turned))

        self.assertEqual(transform, self.transform_for(upright))

    def test_no_frame_with_a_wcs_falls_back_for_all_of_them(self):
        with self.assertLogs(level='WARNING'):
            self.assertIsNone(_stack_orientation(self.stack({}, {}, {})))

    def test_an_unreadable_frame_contributes_nothing(self):
        '''The scaling loop opens the same file next and reports the real failure.'''
        cd = lco_cd(180.0, True, False)
        frames = self.stack(cd, cd) + [self.path('nope.fits')]

        self.assertEqual(_stack_orientation(frames), self.transform_for(cd))


class TestColourStackOrientation(ConversionTestCase):

    def stack(self, *cds):
        return [write_fits(self.path('{}.fits'.format(i)), cd, naxis=256)
                for i, cd in enumerate(cds)]

    def test_the_stack_is_oriented_from_its_wcs(self):
        cd = lco_cd(180.0, True, False)
        out = self.path('wcs.jpg')

        self.assertTrue(fits_to_img(self.stack(cd, cd, cd), out, 'jpeg',
                                    width=64, height=64, color=True))

        _, y = brightest_pixel(Image.open(out))
        self.assertLess(y, 32, 'north did not end up in the top half')

    def test_a_frame_without_a_wcs_is_oriented_with_the_rest(self):
        cd = lco_cd(180.0, True, False)
        out = self.path('mixed.jpg')

        self.assertTrue(fits_to_img(self.stack(cd, cd, {}), out, 'jpeg',
                                    width=64, height=64, color=True))

        _, y = brightest_pixel(Image.open(out))
        self.assertLess(y, 32, 'north did not end up in the top half')

    def test_a_stack_with_no_wcs_anywhere_still_converts(self):
        out = self.path('nowcs.jpg')

        with self.assertLogs(level='WARNING'):
            self.assertTrue(fits_to_img(self.stack({}, {}, {}), out, 'jpeg',
                                        width=64, height=64, color=True))

        self.assertEqual(Image.open(out).mode, 'RGB')

    def test_a_missing_frame_still_returns_false(self):
        cd = lco_cd(180.0, True, False)
        frames = self.stack(cd, cd) + [self.path('nope.fits')]

        self.assertFalse(fits_to_img(frames, self.path('out.jpg'), 'jpeg', color=True))


class TestMultiFitsOrientation(ConversionTestCase):
    """The composite path resolves its orientation the same way the colour stack does."""

    def channel(self, path, colour=(1, 1, 1)):
        return dict(fits_path=path, scale_algorithm='zscale', color=colour,
                    zmin=900, zmax=60000)

    def composite(self, out, inputs):
        self.assertTrue(multi_fits_to_img(inputs, out, width=64, height=64))
        return brightest_pixel(Image.open(out))

    def test_quick_scale_image_stashes_the_frames_transform(self):
        cd = lco_cd(180.0, True, False)
        channel = self.channel(write_fits(self.path('f.fits'), cd, naxis=256))

        quick_scale_image(channel)

        self.assertEqual(channel['transform'], orientation_transform(header_with_cd(cd)))

    def test_north_ends_up_in_the_top_half(self):
        cd = lco_cd(180.0, True, False)
        inputs = [self.channel(write_fits(self.path('{}.fits'.format(i)), cd, naxis=256))
                  for i in range(3)]

        _, y = self.composite(self.path('out.jpg'), inputs)

        self.assertLess(y, 32, 'north did not end up in the top half')

    def test_instruments_at_different_rotations_agree(self):
        positions = []
        for rotation in (0.0, 90.0, 180.0, 270.0):
            path = write_fits(self.path('rot{}.fits'.format(int(rotation))),
                              lco_cd(rotation, True, False), naxis=256)
            positions.append(self.composite(self.path('rot{}.jpg'.format(int(rotation))),
                                            [self.channel(path)]))

        for position in positions[1:]:
            self.assert_near(positions[0], position)

    def test_an_input_given_as_data_takes_the_orientation_of_its_siblings(self):
        """Datalab mixes paths and raw arrays, and an array carries no header to read."""
        cd = lco_cd(180.0, True, False)
        path = write_fits(self.path('f.fits'), cd, naxis=256)
        data, _ = get_reduced_dimensionality_data(path)
        inputs = [self.channel(path),
                  dict(fits_data=data, scale_algorithm='zscale', color=(1, 1, 1),
                       zmin=900, zmax=60000)]

        _, y = self.composite(self.path('mixed.jpg'), inputs)

        self.assertLess(y, 32, 'north did not end up in the top half')

    def test_a_channel_no_input_contributes_to_stays_zero(self):
        """sum blends onto the output buffer, so it has to start zeroed, not just allocated."""
        cd = lco_cd(0.0, True, False)
        inputs = [self.channel(write_fits(self.path('r.fits'), cd, naxis=128), (1, 0, 0)),
                  self.channel(write_fits(self.path('g.fits'), cd, naxis=128), (0, 1, 0))]
        out = self.path('out.tiff')

        # Leave a dirty block the size the blend is about to ask for. numpy serves a
        # request this size from its own freed memory rather than from a fresh zeroed page.
        dirty = np.full((128, 128, 3), 12345.0, dtype=np.float32)
        del dirty

        multi_fits_to_img(inputs, out, blending_algorithm='sum', file_type='tiff',
                          width=128, height=128)

        blue = np.asarray(Image.open(out))[:, :, 2]
        self.assertEqual(blue.max(), 0, 'the blue channel picked up what was left in memory')

    def test_no_input_with_a_wcs_falls_back_to_the_vertical_flip(self):
        inputs = [self.channel(write_fits(self.path('{}.fits'.format(i)), {}, naxis=256))
                  for i in range(3)]

        with self.assertLogs(level='WARNING'):
            self.assertTrue(multi_fits_to_img(inputs, self.path('out.jpg'),
                                              width=64, height=64))


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
