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

        In order to get the four points around a point, we need to floor the values.

        Note: Why not just convert to an integer? A: Because floor vs int are different when dealing
        with negative number. Int will return the number closest to 0, for example, -7.24
        will be converted into -7. Floor on the other hand, gets the next lowest number, so -7.24 gets
        converted into -8. Using int vs floor will return very different results.

        For example, (5.667, 4.563) will become (5, 4). In order to preform bilinear interpolation, we
        must get at least 4 coordinates around (5, 4). In this case, those coordinates would be the following:
        [(5, 4), (6, 4), (5, 5), (6, 5)].

        Next, we need to get the value of the field at those coordinates. eg. field[5, 4],
        field[5, 5], field[6, 4] and field[6, 5]. Those will be the values we are going to be interpolating.
        We then preform bilinear interpolation on the 4 returned values by preforming linear
        interpolation on the bottom two coordinates, field[5, 4] and field[5, 5].
        We then do the same for the other two. (field[6, 4], field[6, 5]).


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

        # lerp(lerp(top_left, top_right, x), lerp(bottom_left, bottom_right, x), y)
        return cls.linear_interpolation(bottom_lerp, top_lerp, padding_y)
