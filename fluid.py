import logging

from pathlib import Path
from typing import Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from interpolation import Interpolation

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
        # self.coordinates: np.ndarray = np.indices(inflow_quantity.shape)
        self.coordinates: np.ndarray = self._build_coordinates(inflow_quantity.shape)

        self.current_timestamp: float = 0.0
        if velocity_field is not None:
            self.velocity_field: np.ndarray = velocity_field.astype(np.float64)
        else:
            self.velocity_field: np.ndarray = np.zeros(self.coordinates.shape)

    def _build_coordinates(self, coordinates_shape: tuple) -> np.ndarray:
        midpoint = [axis // 2 for axis in coordinates_shape]  # x // 2, y // 2
        return np.mgrid[-midpoint[0]:midpoint[0], -midpoint[1]:midpoint[1]]

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

    def advect(self, field: np.ndarray, indices: np.ndarray, timestep: float) -> np.ndarray:
        # This uses the backwards Euler method. (implicit Euler)
        # Timestamp is dynamic, it is changed constantly, but timestep is a constant.
        # self.coordinates is also a constant.
        delta_t = timestep * np.min(self.inflow_quantity.shape)  # Some speedup in simulation time
        advection_coordinates = self.coordinates - (indices * delta_t)

        # logger.info(f"Indices:\n{indices}\nField:\n{advection_coordinates}")

        # Bilerp the velocity fields, then apply the difference to the field.
        return Interpolation.bilinear_interpolation(field, advection_coordinates)

    def step(self, timestep: float = 1 / 240) -> Tuple[np.ndarray, np.ndarray]:
        # timestep = dt (delta T)
        # U = velocity field full of x coordinates
        # V = velocity field full of y coordinates
        # D = density, or the color map. Here called self.inflow_quantity
        logger.info(f"Current timestamp: {self.current_timestamp:.4f}.")
        # print("Velocity Field at initial position (5, 5):")
        # print(self.velocity_field[0, x, y], self.velocity_field[1, x, y])
        # x, y = 3 + (self.velocity_field.shape[1] // 2), 3 + (self.velocity_field.shape[2] // 2)
        # logger.info("\x1b[0;36mVelocity Field at initial position (3, 3):\x1b[0m")
        # logger.info(f"\x1b[0;36mConverted Coordinates: ({x}, {y})\x1b[0m")
        # logger.info(f"\x1b[0;36m({self.velocity_field[0, x, y]:.4f}, {self.velocity_field[1, x, y]:.4f})\x1b[0m")

        u_velocity, v_velocity = self.velocity_field.astype(np.float64)
        # u_velocity, v_velocity = self.enforce_velocity_boundaries(u_velocity, v_velocity)
        advected_u = self.advect(u_velocity, self.velocity_field, timestep)
        advected_v = self.advect(v_velocity, self.velocity_field, timestep)
        # logger.info(f"Velocity U:\n{u_velocity}\nVelocity V:\n{v_velocity}")
        # logger.info(f"Advected U:\n{advected_u}\nAdvected V:\n{advected_v}")
        # comment out velocity_field update line to stop advecting velocity field. DEBUG!
        self.velocity_field = np.array([advected_u, advected_v])
        self.inflow_quantity = self.advect(self.inflow_quantity, self.velocity_field, timestep)
        # logger.debug(f"Advected Color Field:\n{self.inflow_quantity}")

        logger.debug(f"Stepping forward with timestep {timestep:.4f}.")
        self.current_timestamp = self.current_timestamp + timestep

        # logger.debug(f"Velocity Field:\n{self.velocity_field}")
        return self.inflow_quantity, self.velocity_field

    def render_fluid(self, output_path: Union[Path, str]) -> None:
        """Renders the current frame from the color field.

        The color field is first clamped to stay within 0 and 255
        converted into a PIL.Image object, and finally saved to the
        given path.

        Args:
            output_path: The location to save the rendered frame to.
        """
        fluid_map = np.clip(self.inflow_quantity, 0, 1)
        image = Image.fromarray(fluid_map * 255).convert("RGB")
        return image.save(output_path)

    def build_plot(self, output_path: Union[Path, str], grid_step: int = 2) -> None:
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
