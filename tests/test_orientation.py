'''Does the WCS orientation actually put north up and east left?

The method is deliberately empirical rather than algebraic: build a frame with a blob
at true north and another at true east, run the transform, and check where they land.
That tests the convention stack end to end - CD matrix, array indexing, and Pillow's
rotation direction - rather than restating the implementation.
'''
import unittest

import numpy as np
from PIL import Image

from fits2image.orientation import (apply_orientation, get_cd_matrix, orient_image,
                                    orientation_ops)
from tests.helpers import EAST, NORTH, brightest_pixel, header_with_cd, lco_cd, make_array

# The four focal_plane.flip.x / flip.y combinations present in site-configuration,
# with the instrument counts they cover as of 2026-08-19.
FLIP_COMBINATIONS = [
    (True, False),   # 102 instruments, incl. every fa*/sq*/ef* checked
    (True, True),    # 38
    (False, False),  # 26
    (False, True),   # 6
]

ROTATIONS = [0.0, 90.0, 180.0, 270.0, 30.0, -119.531]


class OrientationTestCase(unittest.TestCase):

    def assert_north_up_east_left(self, cd, ops, naxis=201):
        '''North must land above centre and east to the left of it.'''
        array = make_array(cd, naxis=naxis, blobs=((NORTH, 255), (EAST, 160)))
        image = apply_orientation(Image.fromarray(array.astype(np.uint8)), ops)

        centre_x, centre_y = image.size[0] / 2, image.size[1] / 2
        north = np.argwhere(np.asarray(image) == 255).mean(axis=0)
        east = np.argwhere(np.asarray(image) == 160).mean(axis=0)

        # argwhere returns (row, col); rows grow downward on screen
        self.assertLess(north[0], centre_y - 20, 'north is not above centre')
        self.assertLess(east[1], centre_x - 20, 'east is not left of centre')


class TestOrientationOps(OrientationTestCase):

    def test_every_instrument_geometry_ends_north_up_east_left(self):
        for flipx, flipy in FLIP_COMBINATIONS:
            for rotation in ROTATIONS:
                with self.subTest(flipx=flipx, flipy=flipy, rotation=rotation):
                    cd = lco_cd(rotation, flipx, flipy)
                    ops = orientation_ops(header_with_cd(cd))
                    self.assert_north_up_east_left(cd, ops)

    def test_rotator_frames_at_arbitrary_sky_angles(self):
        '''A 2m rotator gives a different sky PA on every exposure.'''
        for sky_pa in (-135.0, -60.0, 0.0, 45.0, 100.0, 175.0):
            with self.subTest(sky_pa=sky_pa):
                cd = lco_cd(0.0, True, False, sky_pa_deg=sky_pa)
                ops = orientation_ops(header_with_cd(cd))
                self.assert_north_up_east_left(cd, ops)

    def test_ops_are_purely_dihedral(self):
        mirror, k = orientation_ops(header_with_cd(lco_cd(90.0, True, False)))
        self.assertIn(mirror, (True, False))
        self.assertIn(k, (0, 1, 2, 3))

    def test_quarter_turn_swaps_the_axes(self):
        '''kb* instruments sit at rotation 90, so their thumbnails change aspect.'''
        image = Image.fromarray(np.zeros((40, 100), dtype=np.uint8))
        self.assertEqual(apply_orientation(image, (False, 1)).size, (40, 100))
        self.assertEqual(apply_orientation(image, (False, 2)).size, (100, 40))

    def test_a_rotation_zero_science_camera_reduces_to_the_legacy_flip(self):
        '''fa11/fa16/sq31 etc: the WCS transform is exactly today's vertical flip,
        which is why 61 of 172 instruments see no change in the central thumbnail.'''
        cd = lco_cd(0.0, True, False)
        self.assertEqual(orientation_ops(header_with_cd(cd)), (True, 2))

        array = np.arange(64, dtype=np.uint8).reshape(8, 8)
        image = Image.fromarray(array)
        wcs = apply_orientation(image, (True, 2))
        legacy = image.transpose(Image.FLIP_TOP_BOTTOM)
        self.assertTrue(np.array_equal(np.asarray(wcs), np.asarray(legacy)))

    def test_mirror_is_applied_before_rotation(self):
        '''The two do not commute, so the order is part of the contract.'''
        array = np.arange(16, dtype=np.uint8).reshape(4, 4)
        image = Image.fromarray(array)
        correct = apply_orientation(image, (True, 1))
        wrong = image.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
        self.assertFalse(np.array_equal(np.asarray(correct), np.asarray(wrong)))


class TestCdMatrix(unittest.TestCase):

    def test_returns_none_without_a_wcs(self):
        self.assertIsNone(get_cd_matrix(None))
        self.assertIsNone(get_cd_matrix({}))
        self.assertIsNone(get_cd_matrix({'CD1_1': 1e-4, 'CD1_2': 0.0}))

    def test_returns_none_for_a_degenerate_or_broken_matrix(self):
        singular = {'CD1_1': 1e-4, 'CD1_2': 1e-4, 'CD2_1': 1e-4, 'CD2_2': 1e-4}
        self.assertIsNone(get_cd_matrix(singular))
        zeroed = {'CD1_1': 0.0, 'CD1_2': 0.0, 'CD2_1': 0.0, 'CD2_2': 0.0}
        self.assertIsNone(get_cd_matrix(zeroed))
        not_a_number = {'CD1_1': float('nan'), 'CD1_2': 0.0, 'CD2_1': 0.0, 'CD2_2': 1e-4}
        self.assertIsNone(get_cd_matrix(not_a_number))
        not_a_float = {'CD1_1': 'unset', 'CD1_2': 0.0, 'CD2_1': 0.0, 'CD2_2': 1e-4}
        self.assertIsNone(get_cd_matrix(not_a_float))

    def test_orientation_ops_returns_none_so_callers_can_fall_back(self):
        self.assertIsNone(orientation_ops({}))


class TestOrientImage(unittest.TestCase):

    def setUp(self):
        self.array = np.arange(64, dtype=np.uint8).reshape(8, 8)
        self.image = Image.fromarray(self.array)

    def test_legacy_applies_the_fixed_vertical_flip(self):
        oriented = orient_image(self.image, header_with_cd(lco_cd(90.0)), orient='legacy')
        self.assertTrue(np.array_equal(np.asarray(oriented), np.flipud(self.array)))

    def test_legacy_honours_flip_v_false(self):
        oriented = orient_image(self.image, None, orient='legacy', flip_v=False)
        self.assertTrue(np.array_equal(np.asarray(oriented), self.array))

    def test_wcs_falls_back_to_the_legacy_flip_without_a_wcs(self):
        with self.assertLogs(level='WARNING'):
            oriented = orient_image(self.image, {}, orient='wcs', flip_v=True)
        self.assertTrue(np.array_equal(np.asarray(oriented), np.flipud(self.array)))

    def test_wcs_ignores_flip_v_when_it_has_a_wcs(self):
        header = header_with_cd(lco_cd(180.0, True, False))
        with_flip = orient_image(self.image, header, orient='wcs', flip_v=True)
        without_flip = orient_image(self.image, header, orient='wcs', flip_v=False)
        self.assertTrue(np.array_equal(np.asarray(with_flip), np.asarray(without_flip)))

    def test_an_unknown_orient_is_rejected(self):
        with self.assertRaises(ValueError):
            orient_image(self.image, None, orient='north-up')


if __name__ == '__main__':
    unittest.main()
