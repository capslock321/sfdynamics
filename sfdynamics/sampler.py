import numpy as np


class FieldSampler(object):
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
    def sample(cls, field: np.ndarray, x: np.ndarray, y: np.ndarray):
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
