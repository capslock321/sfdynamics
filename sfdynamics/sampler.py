import numpy as np


class FieldSampler(object):
    @classmethod
    def clamp(cls, field: np.ndarray, maximum: float) -> np.ndarray:
        """Clamps the given field.

        This really isn't clamping, but rather integer overflowing.

        Args:
            field: The given field to clamp.
            maximum: The maximum value before overflowing.

        Returns:
            np.ndarray: The clamped array.
        """
        return (field + maximum) % maximum

    @classmethod
    def sample(cls, field: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Gets the current value in a field given an array of x and y coordinates.

        Args:
            field: The field to get the values from given x and y.
            x: An array of x coordinates.
            y: An array of y coordinates.

        Returns:
            np.ndarray: An array of values from the given x and y coordinates.
        """
        x_coordinates = x + np.abs(-field.shape[0] // 2)
        y_coordinates = y + np.abs(-field.shape[1] // 2)

        bounded_x = cls.clamp(x_coordinates, field.shape[0])
        bounded_y = cls.clamp(y_coordinates, field.shape[1])
        return field[bounded_x, bounded_y]
