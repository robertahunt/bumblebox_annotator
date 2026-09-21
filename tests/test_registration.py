"""Synthetic registration tests requiring only NumPy and OpenCV."""

import unittest

import cv2
import numpy as np

from core.registration import (
    common_landmark_indices,
    fit_landmark_homography,
    landmark_array,
    registration_error_diagnostics,
)


def project(points, matrix):
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(1, -1, 2), matrix
    ).reshape(-1, 2)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.array(
            [[0, 0], [80, 0], [80, 60], [0, 60],
             [20, 20], [60, 20], [60, 40], [20, 40]],
            dtype=np.float32,
        )
        self.labels = tuple(f"point_{index}" for index in range(len(self.reference)))

    def test_obscured_landmarks_preserve_labels_and_intersect_visibility(self):
        source_points = self.reference.tolist()
        destination_points = self.reference.tolist()
        source_points[1] = None
        destination_points[6] = None
        source = landmark_array(source_points, self.labels)
        destination = landmark_array(destination_points, self.labels)
        indices = common_landmark_indices(source, destination, range(8))
        np.testing.assert_array_equal(indices, [0, 2, 3, 4, 5, 7])
        self.assertTrue(np.isnan(source[1]).all())
        matrix = fit_landmark_homography(source[indices], destination[indices])
        np.testing.assert_allclose(project(self.reference, matrix), self.reference, atol=1e-4)

    def test_invalid_landmarks_are_rejected_before_fitting(self):
        cases = [
            ([[1, 2]], ["a", "b"]),
            ([[1, 2], [1, 2]], ["a", "b"]),
            ([[float("nan"), 2]], ["a"]),
            ([[1, 2, 3]], ["a"]),
        ]
        for points, labels in cases:
            with self.subTest(points=points), self.assertRaises(ValueError):
                landmark_array(points, labels)

    def test_perspective_fit_maps_source_into_reference_coordinates(self):
        motion = np.array(
            [[1.1, 0.12, 20], [-0.05, 0.9, 8], [0.001, 0.002, 1]], dtype=float
        )
        source = project(self.reference, motion)
        matrix = fit_landmark_homography(source, self.reference)
        np.testing.assert_allclose(project(source, matrix), self.reference, atol=1e-4)

    def test_three_point_affine_fallback_is_explicit(self):
        reference = self.reference[:3]
        motion = np.array([[1.2, 0.15, 11], [-0.05, 0.8, -4], [0, 0, 1]], dtype=float)
        source = project(reference, motion)
        with self.assertRaises(ValueError):
            fit_landmark_homography(source, reference)
        matrix = fit_landmark_homography(source, reference, allow_affine=True)
        np.testing.assert_allclose(project(source, matrix), reference, atol=1e-4)
        np.testing.assert_array_equal(matrix[2], [0, 0, 1])

    def test_four_point_fit_preserves_perspective_with_affine_fallback_enabled(self):
        reference = self.reference[:4]
        motion = np.array([[1, 0.1, 10], [0.05, 1, 20], [0.002, 0.001, 1]], dtype=float)
        source = project(reference, motion)
        matrix = fit_landmark_homography(source, reference, allow_affine=True)
        np.testing.assert_allclose(project(source, matrix), reference, atol=1e-4)
        self.assertGreater(np.linalg.norm(matrix[2, :2]), 0.0001)

    def test_independent_landmark_groups_can_have_different_motion(self):
        reference = self.reference.copy()
        reference[4:] += [200, 100]
        source = reference.copy()
        source[:4] += [20, 5]
        source[4:] += [40, -10]
        for indices, shift in ((range(4), [20, 5]), (range(4, 8), [40, -10])):
            selected = common_landmark_indices(source, reference, indices)
            matrix = fit_landmark_homography(source[selected], reference[selected])
            np.testing.assert_allclose(matrix[:2, 2], -np.array(shift), atol=1e-4)

    def test_bad_click_is_rejected_and_reported_by_label(self):
        source = self.reference + [20, 5]
        source[-1] = [150, 300]
        source = landmark_array(source, self.labels)
        matrix = fit_landmark_homography(source, self.reference)
        np.testing.assert_allclose(
            project(source[:-1], matrix), self.reference[:-1], atol=1e-4
        )
        diagnostics = registration_error_diagnostics(
            source, self.reference, matrix, range(8), self.labels
        )
        self.assertEqual(diagnostics["outlier_labels"], ["point_7"])
        self.assertEqual(diagnostics["inlier_count"], 7)
        self.assertGreater(diagnostics["errors_by_label"]["point_7"], 10)

    def test_diagnostics_report_obscured_points_without_counting_them_as_errors(self):
        source_points = self.reference.tolist()
        source_points[3] = None
        source = landmark_array(source_points, self.labels)
        diagnostics = registration_error_diagnostics(
            source, self.reference, np.eye(3), range(8), self.labels
        )
        self.assertEqual(diagnostics["obscured_labels"], ["point_3"])
        self.assertEqual(diagnostics["landmark_count"], 7)
        self.assertEqual(diagnostics["expected_landmark_count"], 8)
        self.assertEqual(diagnostics["rmse"], 0)
        self.assertNotIn("point_3", diagnostics["errors_by_label"])

    def test_no_visible_points_have_unknown_error_not_zero_error(self):
        source = landmark_array([None] * 8, self.labels)
        diagnostics = registration_error_diagnostics(
            source, self.reference, np.eye(3), range(8), self.labels
        )
        self.assertIsNone(diagnostics["rmse"])
        self.assertEqual(diagnostics["inlier_count"], 0)
        self.assertEqual(diagnostics["obscured_labels"], list(self.labels))

    def test_degenerate_coordinates_do_not_produce_a_usable_transform(self):
        line = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32)
        for count, allow_affine in ((3, True), (4, True), (4, False)):
            with self.subTest(count=count, allow_affine=allow_affine):
                with self.assertRaises(ValueError):
                    fit_landmark_homography(
                        line[:count], line[:count], allow_affine=allow_affine
                    )

    def test_malformed_correspondences_and_thresholds_are_rejected(self):
        for source in (self.reference[:3], [[1, 2, 3]], np.full((8, 2), np.nan)):
            with self.subTest(source=source), self.assertRaises(ValueError):
                fit_landmark_homography(source, self.reference)
        for threshold in (0, -1, float("inf"), float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                fit_landmark_homography(
                    self.reference, self.reference, ransac_threshold_px=threshold
                )


if __name__ == "__main__":
    unittest.main()
