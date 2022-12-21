import logging

import numpy as np

logger = logging.getLogger(__name__)


class Interpolation(object):

    @classmethod
    def linear_interpolation(cls, a, b, x) -> np.ndarray:  # also called lerp
        """Implementation adapted from https://en.wikipedia.org/wiki/Linear_interpolation"""
        return a * (1 - x) + x * b

    @classmethod
    def clamp(cls, field: np.ndarray, maximum: float) -> np.ndarray:
        """Clamps the given field.

        This really isn't "clamping", but rather overflowing. To explain,
        when a value is over the maximum given in the maximum parameter
        the value will actually go back to 0, plus the remainder that
        was left over after the overflow.

        Use np.clip(field, 0, field.shape[0] - 1) and it's y-axis
        equivalent np.clip(field, 0, field.shape[1] - 1) if you do
        not want this behavior.

        Args:
            field: The given field to clamp.
            maximum: The maximum value before overflowing.

        Returns:
            np.ndarray: The clamped array.
        """
        return (field + maximum) % maximum

    @classmethod
    def value_at(cls, field: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Gets the current value in a field given an array of x and y coordinates.

        The given field is clamped to prevent overflow and underflow. We subtract
        one from the field's y shape in order for the field to be indexed properly.
        EG. If you have an array ["one", "two", "three"] you would not index get
        the value "three" using array[3], but rather using array[2].

        What is x and y being added with? Since the coordinate field is negative
        we need to get the lowest value in the array, for example -16 and add 16 to that
        so then valid indices are created. We don't want to index with negative numbers
        because that will instead get the incorrect value. Think trying to get the value
        at -2, but actually getting the value at 14.

        Example: We have an array that ranges from -16 to 16. If we add 16 to that
        we get a range from 0 to 32, which can be indexed properly.

        Args:
            field: The field to get the values from given x and y.
            x: An array of x coordinates.
            y: An array of y coordinates.

        Returns:
            np.ndarray: An array of values from the given x and y coordinates.
        """
        x_coordinates = x + np.abs(-field.shape[0] // 2)
        y_coordinates = y + np.abs(-field.shape[1] // 2)
        # What the hell does ^^^ do? It simply makes a negative number into a positive one.
        bounded_x = cls.clamp(x_coordinates, field.shape[0])
        bounded_y = cls.clamp(y_coordinates, field.shape[1])
        # bounded_x = np.clip(x_coordinates, 0, field.shape[0] - 1)
        # bounded_y = np.clip(y_coordinates, 0, field.shape[1] - 1)
        return field[bounded_x, bounded_y]

    @classmethod
    def bilinear_interpolation(cls, field: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """Preform bilinear interpolation on the given field, given the velocity field.

        In order to get the four points around a point, we need to floor the values.

        Note: Why not just convert to an integer? A: Because floor vs int are different when dealing
        with negative numbers. Using int vs floor will return very different results. % 1 is used to ease the transition
        between negative and positive numbers.

        For example, (5.667, 4.563) will become (5, 4). Then we get the four points around (5, 4). Using
        those four points, we get the values from field at those four coordinates. Finally, we lerp the four values
        and put the resulting number (should be between 0 and 1) into the new field.

        Args:
            field: The color field. This field is what is converted into the final image.
            coordinates: A list of new advected positions.

        Returns:
            np.ndarray: The new color field.
        """
        # coordinates = np.array([coordinates[0] - padding[0], coordinates[1] - padding[1]])
        coordinates_x, coordinates_y = np.floor(coordinates).astype(int)  # This floors the values.

        # coordinates = np.array([coordinates[0] - padding[0], coordinates[1] - padding[1]])

        padding_x = np.clip(coordinates[0] - coordinates_x, 0, 1)
        padding_y = np.clip(coordinates[1] - coordinates_y, 0, 1)

        top_left = cls.value_at(field, coordinates_x + 0, coordinates_y + 0)  # x00
        top_right = cls.value_at(field, coordinates_x + 1, coordinates_y + 0)  # x10
        bottom_left = cls.value_at(field, coordinates_x + 0, coordinates_y + 1)  # x01
        bottom_right = cls.value_at(field, coordinates_x + 1, coordinates_y + 1)  # x11

        # top_left is the default position
        # logger.debug(f"Coordinates X:\n{coordinates[0]}\nCoordinates Y:\n{coordinates[1]}")
        # logger.debug(f"Rounded X:\n{coordinates_x}\nRounded Y:\n{coordinates_y}")
        # logger.debug(f"Top Right:\n{top_right}\nBottom Right:\n{bottom_right}")
        # logger.debug(f"Top Left:\n{top_left}\nBottom Left:\n{bottom_left}")
        # logger.debug(f"Padding X:\n{padding_x}\nPadding Y:\n{padding_y}")

        # 0.5 samples the color at the middle of the cell rather than bottom right
        # But for velocity, aka U and V, the padding must be (0.0, 0.5) and (0.5, 0.0) respectively.
        top_lerp = cls.linear_interpolation(top_left, top_right, padding_x)
        bottom_lerp = cls.linear_interpolation(bottom_left, bottom_right, padding_x)

        # logger.debug(f"Top Lerp:\n{top_lerp}\nBottom Lerp:\n{bottom_lerp}")
        return cls.linear_interpolation(top_lerp, bottom_lerp, padding_y)
