'''Reading the frame: HDU selection, header merging, and the scaling split.'''
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from fits2image import scaling
from fits2image.scaling import (auto_scale, auto_scale_data, get_frame_header,
                                get_reduced_dimensionality_data, get_scaled_image,
                                least_squares_line_fit)
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

    def test_the_header_only_read_agrees_with_the_full_one(self):
        '''fits_to_img uses this to pick one orientation for a whole colour stack.'''
        path = write_multi_hdu_fits(self.path('sinistro.fits'), lco_cd(180.0, True, False))

        _, from_full_read = get_reduced_dimensionality_data(path)

        self.assertEqual(get_frame_header(path).tostring(), from_full_read.tostring())

    def test_an_hdu_with_a_zero_length_axis_is_skipped(self):
        '''A (64, 0) HDU has two dimensions and no pixels.'''
        from astropy.io import fits
        path = self.path('thin.fits')
        fits.HDUList([fits.PrimaryHDU(data=np.zeros((64, 0), dtype=np.float32)),
                      fits.ImageHDU(data=np.ones((32, 32), dtype=np.float32))]).writeto(path)

        data, _ = get_reduced_dimensionality_data(path)

        self.assertEqual(data.shape, (32, 32))

    def test_a_table_extension_is_skipped(self):
        '''BANZAI frames carry a CAT extension alongside the science array.'''
        from astropy.io import fits
        path = self.path('with_cat.fits')
        catalogue = fits.BinTableHDU.from_columns(
            [fits.Column(name='x', format='E', array=np.arange(10.0))], name='CAT')
        fits.HDUList([fits.PrimaryHDU(), catalogue,
                      fits.ImageHDU(data=np.ones((32, 32), dtype=np.float32), name='SCI')]).writeto(path)

        data, _ = get_reduced_dimensionality_data(path)

        self.assertEqual(data.shape, (32, 32))

    def test_a_frame_with_no_image_data_raises(self):
        from astropy.io import fits
        path = self.path('empty.fits')
        fits.PrimaryHDU().writeto(path)

        with self.assertRaises(Exception):
            get_reduced_dimensionality_data(path)


class TestLineFit(unittest.TestCase):

    def test_a_rank_deficient_fit_still_produces_an_rms(self):
        '''lstsq returns an empty residual array when the fit is rank deficient, and
        numpy 2 will not convert that to a float, so the residual is computed directly.'''
        slope, y_intercept, iterations, nfitsamples, rms, samples = least_squares_line_fit(np.array([5.0]))

        self.assertEqual(nfitsamples, 1)
        self.assertEqual(y_intercept, 5.0)
        self.assertEqual(rms, 0.0)

    def test_a_full_rank_fit_uses_the_residual_lstsq_returns(self):
        straight_line = np.arange(100, dtype=float)

        slope, y_intercept, iterations, nfitsamples, rms, samples = least_squares_line_fit(straight_line)

        self.assertAlmostEqual(slope, 1.0)
        self.assertAlmostEqual(rms, 0.0)


class TestAutoScaleSplit(ScalingTestCase):

    def test_auto_scale_data_matches_auto_scale(self):
        path = write_fits(self.path('frame.fits'), lco_cd(), naxis=128)

        from_path = auto_scale(path)
        data, header = get_reduced_dimensionality_data(path)
        from_data = auto_scale_data(data, header)

        self.assertTrue(np.array_equal(from_path, from_data))


class TestGetScaledImage(ScalingTestCase):

    def test_the_frame_is_only_read_once(self):
        '''Orientation needs the header and scaling needs the data. Both come out of one
        open, and the thumbnail service holds a whole frame in memory per message, so
        this guards the split against growing a second read of a 4k x 4k file.'''
        path = write_fits(self.path('frame.fits'), lco_cd(), naxis=128)

        real = scaling.get_reduced_dimensionality_data
        with mock.patch.object(scaling, 'get_reduced_dimensionality_data',
                               side_effect=real) as reader:
            get_scaled_image(path)

        self.assertEqual(reader.call_count, 1)

    def test_a_zmin_and_zmax_of_zero_are_not_treated_as_unset(self):
        '''An explicit pair of zeros is a linear scale, not a request to auto scale.'''
        path = write_fits(self.path('frame.fits'), lco_cd(), naxis=128)

        with mock.patch.object(scaling, 'auto_scale_data') as auto_scale:
            get_scaled_image(path, zmin=0, zmax=0)

        auto_scale.assert_not_called()

    def test_a_half_specified_pair_auto_scales(self):
        '''linear_scale does arithmetic on both limits, so one on its own is no use.'''
        path = write_fits(self.path('frame.fits'), lco_cd(), naxis=128)

        with mock.patch.object(scaling, 'auto_scale_data',
                               side_effect=scaling.auto_scale_data) as auto_scale:
            get_scaled_image(path, zmin=900)

        auto_scale.assert_called_once()

    def test_a_rotation_180_instrument_moves_off_the_plain_flip(self):
        '''fa14, fa01, ef14 and the rest of the 180 degree group.'''
        path = write_fits(self.path('fa14.fits'), lco_cd(180.0, True, False), naxis=128)

        oriented = np.asarray(get_scaled_image(path))
        flipped = np.asarray(get_scaled_image(path, transform=None))

        self.assertFalse(np.array_equal(oriented, flipped))

    def test_a_rotation_0_instrument_keeps_the_plain_flip(self):
        '''fa11, fa16, sq31 - already correct today, so they must not move.'''
        path = write_fits(self.path('fa11.fits'), lco_cd(0.0, True, False), naxis=128)

        oriented = np.asarray(get_scaled_image(path))
        flipped = np.asarray(get_scaled_image(path, transform=None))

        self.assertTrue(np.array_equal(oriented, flipped))

    def test_a_frame_without_a_wcs_falls_back_to_the_flip(self):
        path = write_fits(self.path('nowcs.fits'), {}, naxis=128)

        with self.assertLogs(level='WARNING') as logged:
            oriented = np.asarray(get_scaled_image(path))
        flipped = np.asarray(get_scaled_image(path, transform=None))

        self.assertTrue(np.array_equal(oriented, flipped))
        # a service converting thousands of frames needs to know which one fell back
        self.assertIn(path, logged.output[0])

    def test_explicit_zmin_zmax_still_orients(self):
        path = write_fits(self.path('fa14.fits'), lco_cd(180.0, True, False), naxis=128)

        oriented = np.asarray(get_scaled_image(path, zmin=900, zmax=2000))
        flipped = np.asarray(get_scaled_image(path, zmin=900, zmax=2000, transform=None))

        self.assertFalse(np.array_equal(oriented, flipped))


if __name__ == '__main__':
    unittest.main()
