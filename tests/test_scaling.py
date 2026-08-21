'''Reading the frame: HDU selection, header merging, and the scaling split.'''
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from fits2image import scaling
from fits2image.scaling import (auto_scale, auto_scale_data, get_reduced_dimensionality_data,
                                get_scaled_image)
from tests.helpers import lco_cd, write_fits, write_multi_hdu_fits


class ScalingTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def path(self, name):
        return os.path.join(self.tmp, name)


class TestHeaderMerging(ScalingTestCase):
    '''Sinistro puts the WCS on the primary HDU and the pixels in four extensions,
    so the header the data came from does not carry the CD matrix at all.'''

    def test_wcs_is_recovered_from_the_primary_header(self):
        cd = lco_cd(180.0, True, False)
        path = write_multi_hdu_fits(self.path('sinistro.fits'), cd)

        data, header = get_reduced_dimensionality_data(path)

        self.assertEqual(data.shape, (64, 64))
        self.assertAlmostEqual(header['CD1_1'], cd['CD1_1'])
        self.assertAlmostEqual(header['CD2_2'], cd['CD2_2'])
        self.assertEqual(header['SATURATE'], 65535)

    def test_the_data_hdu_still_wins_for_its_own_keywords(self):
        '''NAXIS must describe the array actually returned, not the empty primary.'''
        path = write_multi_hdu_fits(self.path('sinistro.fits'), lco_cd(), naxis=32)

        _, header = get_reduced_dimensionality_data(path)

        self.assertEqual(header['NAXIS1'], 32)
        self.assertEqual(header['NAXIS2'], 32)
        self.assertEqual(header['BITPIX'], -32)

    def test_a_single_hdu_frame_skips_the_merge(self):
        '''Site flash frames are single-HDU, so merging would only duplicate
        COMMENT and HISTORY cards onto themselves.'''
        path = write_fits(self.path('flash.fits'), lco_cd(), naxis=64)

        from astropy.io import fits
        with fits.open(path) as hdul:
            expected = len(hdul[0].header)

        _, header = get_reduced_dimensionality_data(path)

        self.assertEqual(len(header), expected)

    def test_a_frame_with_no_image_data_raises(self):
        from astropy.io import fits
        path = self.path('empty.fits')
        fits.PrimaryHDU().writeto(path)

        with self.assertRaises(Exception):
            get_reduced_dimensionality_data(path)


class TestAutoScaleSplit(ScalingTestCase):

    def test_auto_scale_data_matches_auto_scale(self):
        path = write_fits(self.path('frame.fits'), lco_cd(), naxis=128)

        from_path = auto_scale(path)
        data, header = get_reduced_dimensionality_data(path)
        from_data = auto_scale_data(data, header)

        self.assertTrue(np.array_equal(from_path, from_data))


class TestGetScaledImage(ScalingTestCase):

    def test_the_frame_is_only_read_once(self):
        '''The thumbnail service holds a whole frame in memory per message, so a
        second read of a 4k x 4k file is not free.'''
        path = write_fits(self.path('frame.fits'), lco_cd(), naxis=128)

        real = scaling.get_reduced_dimensionality_data
        with mock.patch.object(scaling, 'get_reduced_dimensionality_data',
                               side_effect=real) as reader:
            get_scaled_image(path)

        self.assertEqual(reader.call_count, 1)

    def test_wcs_and_legacy_differ_for_a_rotation_180_instrument(self):
        '''fa14, fa01, ef14 and the rest of the 180 degree group.'''
        path = write_fits(self.path('fa14.fits'), lco_cd(180.0, True, False), naxis=128)

        legacy = np.asarray(get_scaled_image(path, orient='legacy'))
        wcs = np.asarray(get_scaled_image(path, orient='wcs'))

        self.assertFalse(np.array_equal(legacy, wcs))

    def test_wcs_and_legacy_agree_for_a_rotation_0_instrument(self):
        '''fa11, fa16, sq31 - already correct today, so they must not move.'''
        path = write_fits(self.path('fa11.fits'), lco_cd(0.0, True, False), naxis=128)

        legacy = np.asarray(get_scaled_image(path, orient='legacy'))
        wcs = np.asarray(get_scaled_image(path, orient='wcs'))

        self.assertTrue(np.array_equal(legacy, wcs))

    def test_wcs_falls_back_to_legacy_without_a_wcs(self):
        path = write_fits(self.path('nowcs.fits'), {}, naxis=128)

        legacy = np.asarray(get_scaled_image(path, orient='legacy'))
        with self.assertLogs(level='WARNING'):
            wcs = np.asarray(get_scaled_image(path, orient='wcs'))

        self.assertTrue(np.array_equal(legacy, wcs))

    def test_explicit_zmin_zmax_still_orients(self):
        path = write_fits(self.path('fa14.fits'), lco_cd(180.0, True, False), naxis=128)

        legacy = np.asarray(get_scaled_image(path, zmin=900, zmax=2000, orient='legacy'))
        wcs = np.asarray(get_scaled_image(path, zmin=900, zmax=2000, orient='wcs'))

        self.assertFalse(np.array_equal(legacy, wcs))


if __name__ == '__main__':
    unittest.main()
