import logging

from pathlib import Path
from typing import Union, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from numpy import ndarray

from .interpolation import Interpolation

logger = logging.getLogger(__name__)


class FluidDynamics(object):
    def __init__(
        self,
        inflow_quantity: np.ndarray,
        initial_velocity: np.ndarray = None,
        advect_velocity: bool = True,
        apply_pressure: bool = True,
    ):
        """A Python implementation of a fluid dynamics solver.

        The inflow_quantity is a 2D array that represents the initial
        dye positions in the image array. You can call this field the color field.

        The equation is used to build the initial velocity field, which will be advected
        by itself on every iteration alongside the color field. The equation
        should be something that can be called, and should take exactly two
        parameters, the first being a np.ndarray containing all x coordinates obtained from
        np.indices, and the other containing all y coordinates.

        Args:
            inflow_quantity: The initial dye positions.
            initial_velocity: The initial velocity field. Can be None.
            advect_velocity: If the velocity field should be advected.
            apply_pressure: If the pressure should be calculated. (Very intensive.)
        """
        self.inflow_quantity: np.ndarray = inflow_quantity
        self.coordinates: np.ndarray = self._build_coordinates(inflow_quantity.shape)

        self.advect_velocity: bool = advect_velocity
        self.apply_pressure: bool = apply_pressure

        if self.apply_pressure:
            self.laplacian_matrix: np.ndarray = self._build_laplacian(2, 0)

        self.current_timestamp: float = 0.0
        if initial_velocity is not None:
            self.velocity_field: np.ndarray = initial_velocity.astype(np.float64)
        else:
            self.velocity_field: np.ndarray = np.zeros(self.coordinates.shape)

    def _build_coordinates(self, coordinates_shape: tuple) -> np.ndarray:
        """Builds the coordinate plane.

        This is used to create the coordinates for the velocity field. The shape of
        the coordinate plane is based off the dye field's shape. The field is split in half
        to allow velocity advection using negative numbers.

        Args:
            coordinates_shape (np.ndarray): The resulting coordinate plane's shape.

        Returns:
            np.ndarray: The built coordinate plane.
        """
        midpoint = [axis // 2 for axis in coordinates_shape]  # x // 2, y // 2
        return np.mgrid[-midpoint[0] : midpoint[0], -midpoint[1] : midpoint[1]]

    def _compute_differences(self, derivative_order: int, accuracy: int) -> tuple[Any, ndarray]:
        """Computes the central finite differences given the derivative order and accuracy.

        This first builds the stencil, which is simply a 1D range of numbers from n to -n, and then constructs
        an 2D array using that stencil. It also constructs a rhs (Right Hand Side), and the resulting linear
        matrix equation is then solved using np.linalg.solve.

        Reference: https://en.wikipedia.org/wiki/Finite_difference_coefficient

        Args:
            derivative_order (int): The derivative order.
            accuracy (int): The accuracy. Used to compute the stencil range.

        Returns:
            np.ndarray: The computed differences.
            np.ndarray: The stencil used to compute the differences.
        """
        stencil_range = accuracy + np.ceil(derivative_order / 2).astype(int)
        laplacian_stencil = np.arange(-stencil_range, stencil_range + 1)
        coefficients = np.flipud(np.vander(laplacian_stencil).transpose())

        rhs = np.zeros((coefficients.shape[0],))
        rhs[derivative_order] = np.math.factorial(derivative_order)
        return np.linalg.solve(coefficients, rhs), laplacian_stencil

    def _build_laplacian(self, derivative: int = 2, accuracy: int = 0) -> np.ndarray:
        """Builds the laplacian matrix.

        This method will only be run once. Using the derivative and accuracy, a laplacian matrix is built
        and repeatedly used by compute_pressure. However, computing the pressure is resource intensive, so
        if pressure is to be computed, the run time will be longer.

        Args:
            derivative (int): The derivative order. Immediately passed into _compute_differences.
            accuracy (int): The accuracy. Immediately passed into _compute_differences.

        Returns:
            np.ndarray: The built laplacian matrix.
        """
        laplacian_matrix = np.zeros((self.inflow_quantity.shape[0],) * 2)
        coefficients, stencil = self._compute_differences(derivative, accuracy)

        for coefficient, offset in zip(coefficients, stencil):
            rows, columns = np.indices(self.inflow_quantity.shape)

            row_diagonal, col_diagonal = np.diag(rows, k=offset), np.diag(columns, k=offset)
            laplacian_matrix[row_diagonal, col_diagonal] = coefficient

        laplacian_eye = np.eye(self.inflow_quantity.shape[0])
        return np.add(np.kron(laplacian_eye, laplacian_matrix), np.kron(laplacian_matrix, laplacian_eye))

    def compute_divergence(self, velocity_field: np.ndarray) -> np.ndarray:
        """Calculates the divergence given the velocity field.

        The divergence is found by adding the gradients of the two velocity fields together.

        Args:
            velocity_field: The velocity field, contains U and V.

        Returns:
            np.ndarray: The added velocity fields, or the divergence.
        """
        u_velocity, v_velocity = velocity_field.astype(np.float64)

        divergence_u = np.gradient(u_velocity, axis=0)
        divergence_v = np.gradient(v_velocity, axis=1)
        return np.add(divergence_u, divergence_v)

    def compute_pressure(self, divergence: np.ndarray) -> np.ndarray:
        """Computes the pressure given the divergence.

        Takes the divergence of the fluid field and solves for the pressure using the static
        laplacian matrix generated on object creation.

        Args:
            divergence (np.ndarray): The divergence of the fluid field.

        Returns:
            np.ndarray: The computed pressure.
        """
        solved_pressure = np.linalg.solve(self.laplacian_matrix, divergence.flatten())
        return np.gradient(solved_pressure.reshape(divergence.shape))

    def advect(self, field: np.ndarray, indices: np.ndarray, delta_t: float) -> np.ndarray:
        """Advects the given field given a list of coordinates.

        This uses the implicit (backwards) Euler method to advect the field. The timestep is multiplied
        by the smallest value in the shape of the dye field to help speed up how fast the
        fluid is advected.

        Args:
            field (np.ndarray): The field to advect. The field requires a 2D array.
            indices (np.ndarray): The coordinates of each particle. This requires a np.indices like shape.
            delta_t (float): The timestep.

        Returns:
            np.ndarray: The advected field. The resulting shape is the same as the field.
        """
        # delta_t = delta_t * np.min(self.inflow_quantity.shape)  # Some speedup in simulation time
        # If simulation seems to be splitting apart, use the timestep directly for more stability
        advection_coordinates = self.coordinates - (indices * delta_t)

        # Bilerp the velocity fields, then apply the difference to the field.
        return Interpolation.bilinear_interpolation(field, advection_coordinates)

    def step(self, timestep: float = 1 / 240) -> Tuple[np.ndarray, np.ndarray]:
        """Steps forward in time.

        dt = timestep, also called Delta T
        U = velocity field full of x coordinates
        V = velocity field full of y coordinates
        D = density, or the color map. Here called self.inflow_quantity

        Args:
            timestep (float): The amount of time to step forward in time.

        Returns:
            np.ndarray: The dye field.
            np.ndarray: The velocity field.
        """
        logger.info(f"Current timestamp: {self.current_timestamp:.4f}.")

        if self.advect_velocity:
            u_velocity, v_velocity = self.velocity_field.astype(np.float64)
            advected_u = self.advect(u_velocity, self.velocity_field, timestep)
            advected_v = self.advect(v_velocity, self.velocity_field, timestep)
            self.velocity_field = np.array([advected_u, advected_v])

        if self.apply_pressure:
            divergence_field = self.compute_divergence(self.velocity_field)
            self.velocity_field -= self.compute_pressure(divergence_field)

        self.inflow_quantity = self.advect(self.inflow_quantity, self.velocity_field, timestep)

        logger.debug(f"Stepping forward with timestep {timestep:.4f}.")
        self.current_timestamp = self.current_timestamp + timestep

        return self.inflow_quantity, self.velocity_field

    def render_fluid(self, output_path: Union[Path, str]) -> None:
        """Renders the current frame from the color field.

        The color field is first clamped to stay within 0 and 1 and
        converted into a PIL.Image object, and finally saved to the
        given path.

        Args:
            output_path: The location to save the rendered frame to.

        """
        fluid_map = np.rollaxis(np.clip(self.inflow_quantity, 0, 1), 1)
        image = Image.fromarray(fluid_map * 255).convert("L")
        return image.save(output_path)

    def build_plot(self, output_path: Union[Path, str], grid_step: int = 2) -> None:
        """Visualizes the velocity field.

        Args:
            output_path (Union[Path, str]): The path to write the resulting graph to.
            grid_step (int): The step size for the velocity field.
        """
        fig, ax = plt.subplots(figsize=(9, 9))

        X, Y = self.coordinates.astype(np.float64)
        U, V = self.velocity_field.astype(np.float64)

        X, Y = X[::grid_step, ::grid_step], Y[::grid_step, ::grid_step]
        U, V = U[::grid_step, ::grid_step], V[::grid_step, ::grid_step]

        ax.quiver(X, Y, U, V, color="red")

        ax.axhline(0, color="black")
        ax.axvline(0, color="black")
        ax.set_aspect("equal")
        ax.grid(color="black")
        plt.savefig(output_path)
        return plt.close(fig)
