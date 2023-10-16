import logging

import numpy as np

from .sampler import FieldSampler

logger = logging.getLogger(__name__)


class Interpolation(object):
    @classmethod
    def linear_interpolation(cls, a, b, x) -> np.ndarray:  # also called lerp
        """Implementation adapted from https://en.wikipedia.org/wiki/Linear_interpolation"""
        return a * (1 - x) + x * b

    @classmethod
    def bilinear_interpolation(cls, field: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """Preform bilinear interpolation on the given field, given the velocity field.

        Args:
            field: The color field. This field is what is converted into the final image.
            coordinates: A list of new advected positions.

        Returns:
            np.ndarray: The new color field.
        """
        coordinates_x, coordinates_y = np.floor(coordinates).astype(int)

        padding_x = np.clip(coordinates[0] - coordinates_x, 0, 1)
        padding_y = np.clip(coordinates[1] - coordinates_y, 0, 1)

        bottom_left = FieldSampler.sample(field, coordinates_x + 0, coordinates_y + 0)  # x00
        bottom_right = FieldSampler.sample(field, coordinates_x + 1, coordinates_y + 0)  # x10
        top_left = FieldSampler.sample(field, coordinates_x + 0, coordinates_y + 1)  # x01
        top_right = FieldSampler.sample(field, coordinates_x + 1, coordinates_y + 1)  # x11

        # 0.5 samples the color at the middle of the cell rather than bottom right
        # But for velocity, aka U and V, the padding must be (0.0, 0.5) and (0.5, 0.0) respectively.
        top_lerp = cls.linear_interpolation(top_left, top_right, padding_x)
        bottom_lerp = cls.linear_interpolation(bottom_left, bottom_right, padding_x)

        return cls.linear_interpolation(bottom_lerp, top_lerp, padding_y)
