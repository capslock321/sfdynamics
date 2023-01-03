import logging

from pathlib import Path
from typing import Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from .sampler import FieldSampler
from .interpolation import Interpolation

logger = logging.getLogger(__name__)


class FluidDynamics(object):
    def __init__(self, inflow_quantity: np.ndarray, velocity_field: np.ndarray = None):
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
            velocity_field: The initial velocity field. Can be None.
        """
        self.inflow_quantity: np.ndarray = inflow_quantity
        self.coordinates: np.ndarray = self._build_coordinates(inflow_quantity.shape)

        # self.advect_velocity: bool = advect_velocity

        self.current_timestamp: float = 0.0
        if velocity_field is not None:
            self.velocity_field: np.ndarray = velocity_field.astype(np.float64)
        else:
            self.velocity_field: np.ndarray = np.zeros(self.coordinates.shape)

        self.divergence_field: np.ndarray = np.zeros(inflow_quantity.shape)  # (HEIGHT, WIDTH)
        self.pressure_field: np.ndarray = np.zeros(inflow_quantity.shape)  # (HEIGHT, WIDTH)

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

    def enforce_velocity_boundaries(self, U: np.ndarray, V: np.ndarray):
        # Very dirty implementation
        U[0] = -U[1]
        U[U.shape[0] - 1] = -U[U.shape[0] - 2]
        V[0] = -V[1]
        V[V.shape[0] - 1] = -V[V.shape[0] - 2]

        U[:, 0] = -U[:, 1]
        U[:, U.shape[1] - 1] = -U[:, U.shape[1] - 2]
        V[:, 0] = -V[:, 1]
        V[:, V.shape[1] - 1] = -V[:, V.shape[1] - 2]
        return U, V

    def enforce_boundaries(self, U: np.ndarray, V: np.ndarray):
        # Very dirty implementation
        U[0] = 0
        U[U.shape[0] - 1] = 0
        V[0] = 0
        V[V.shape[0] - 1] = 0

        U[:, 0] = 0
        U[:, U.shape[1] - 1] = 0
        V[:, 0] = 0
        V[:, V.shape[1] - 1] = 0
        return U, V

    def compute_divergence(self, velocity_field: np.ndarray) -> np.ndarray:
        u_velocity, v_velocity = velocity_field.astype(np.float64)

        divergence_u = np.gradient(u_velocity, axis=0)
        divergence_v = np.gradient(v_velocity, axis=1)
        return np.add(divergence_u, divergence_v)

    def compute_pressure(self, field: np.ndarray, rhs: np.ndarray, max_iterations: int = 100):
        ...

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

    def jacobi(self, alpha: int = -1, beta: float = 0.25, max_iterations: int = 16):
        """
        for(i = 0; i < iterations; i++) {
            for(var y = 1; y < HEIGHT-1; y++) {
                for(var x = 1; x < WIDTH-1; x++) {
                    var x0 = p0(x-1, y),
                        x1 = p0(x+1, y),
                        y0 = p0(x, y-1),
                        y1 = p0(x, y+1);
                    p1(x, y, (x0 + x1 + y0 + y1 + alpha * b(x, y)) * beta);
                }
        }
        """
        x_coordinates, y_coordinates = self.coordinates.astype(int)
        for iteration in range(max_iterations):
            x0 = FieldSampler.sample(self.pressure_field, x_coordinates - 1, y_coordinates - 0)
            x1 = FieldSampler.sample(self.pressure_field, x_coordinates + 1, y_coordinates + 0)
            y0 = FieldSampler.sample(self.pressure_field, x_coordinates - 0, y_coordinates - 1)
            y1 = FieldSampler.sample(self.pressure_field, x_coordinates + 0, y_coordinates + 1)
            divergence = FieldSampler.sample(self.divergence_field, x_coordinates, y_coordinates)
            self.pressure_field = (x0 + x1 + y0 + y1 + alpha * divergence) * beta
        return self.pressure_field

    def subtract_pressure(self, velocity_field: np.ndarray):
        """
        function subtractPressureGradient(ux, uy, p){
            for(var y = 1; y < HEIGHT-1; y++) {
                for(var x = 1; x < WIDTH-1; x++) {
                    var x0 = p(x-1, y),
                        x1 = p(x+1, y),
                        y0 = p(x, y-1),
                        y1 = p(x, y+1),
                        dx = (x1-x0)/2,
                        dy = (y1-y0)/2;
                        ux(x, y, ux(x, y)-dx);
                        uy(x, y, uy(x, y)-dy);
                }
            }
        }
        """
        x_coordinates, y_coordinates = self.coordinates.astype(int)
        x0 = FieldSampler.sample(self.pressure_field, x_coordinates - 1, y_coordinates - 0)
        x1 = FieldSampler.sample(self.pressure_field, x_coordinates + 1, y_coordinates + 0)
        y0 = FieldSampler.sample(self.pressure_field, x_coordinates - 0, y_coordinates - 1)
        y1 = FieldSampler.sample(self.pressure_field, x_coordinates + 0, y_coordinates + 1)
        dx, dy = (x1 - x0) / 2, (y1 - y0) / 2
        return velocity_field[0] - dx, velocity_field[1] - dy

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

        u_velocity, v_velocity = self.velocity_field.astype(np.float64)
        advected_u = self.advect(u_velocity, self.velocity_field, timestep)
        advected_v = self.advect(v_velocity, self.velocity_field, timestep)
        self.velocity_field = np.array([advected_u, advected_v])

        self.divergence_field = self.compute_divergence(self.velocity_field)
        self.jacobi(-1, 0.25, max_iterations=100)
        self.velocity_field = np.array(self.subtract_pressure(self.velocity_field))
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
        image = Image.fromarray(fluid_map * 255).convert("RGB")
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
