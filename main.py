import os
import sys
import time
import shutil
import logging

from typing import Union
from pathlib import Path

import scipy
import imageio
import matplotlib
import numpy as np

from sfdynamics import FluidDynamics

logger = logging.getLogger(__name__)

fmt = "\x1b[32m[%(asctime)s] %(name)s: %(message)s\x1b[0m"
logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%m/%d/%Y %I:%M:%S %p")

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

matplotlib.use("Agg")  # Pycharm debugging purposes.

logging.getLogger("matplotlib").setLevel(logging.WARNING)  # Debug
logging.getLogger("PIL").setLevel(logging.WARNING)  # Debug

# RESOLUTION = 2, 2
# RESOLUTION = 4, 4
RESOLUTION = 8, 8
# RESOLUTION = 16, 16
# RESOLUTION = 32, 32  # test resolution
# RESOLUTION = 64, 64  # PROD Resolution
# RESOLUTION = 128, 128
# RESOLUTION = 256, 256

# Creates a checkerboard pattern
inflow_dye = np.indices(RESOLUTION).sum(axis=0) % 2
inflow_dye = np.kron(inflow_dye, np.ones((16, 16)))
inflow_dye = inflow_dye.astype(np.uint8)

# REMEMBER, FOR STUFF LIKE SIN AND COS, CONVERT TO DEGREES!

"""
def equation(indices: np.ndarray):
    # multiple circles, extremely weird map
    x, y = indices.astype(np.float64)
    X_field = np.sin(x) + np.sin(y)
    Y_field = np.sin(x) - np.sin(y)
    return np.array([X_field, Y_field])


def equation(indices: np.ndarray, step: int):
    # multiple circles, extremely weird map
    x, y = indices.astype(np.float64)
    x, y = x[::step, ::step], y[::step, ::step]
    X_field = scipy.ndimage.zoom(np.sin(np.rad2deg(np.pi * y)), step)
    Y_field = scipy.ndimage.zoom(np.sin(np.rad2deg(np.pi * x)), step)
    return np.array([X_field, Y_field])


def equation(indices: np.ndarray) -> np.ndarray:
    # high tendency to overflow
    x, y = indices.astype(np.float64)
    x_field = x ** 2 - y ** 2 - 4
    y_field = 2 * x * y
    return np.array([x_field, y_field])


def equation(indices: np.ndarray, step: int = 16) -> np.ndarray:
    x, y = indices.astype(np.float64)
    x, y = x[::step, ::step], y[::step, ::step]
    X_field = scipy.ndimage.zoom(-y, step)
    Y_field = scipy.ndimage.zoom(x, step)
    return np.array([X_field, Y_field])

def equation(indices: np.ndarray):
    x, y = indices.astype(np.float64)
    return np.array([np.rad2deg(2*np.pi*x), np.zeros(y.shape)])
"""


def equation(indices: np.ndarray, step: int = 16) -> np.ndarray:
    x, y = indices.astype(np.float64)
    x, y = x[::step, ::step], y[::step, ::step]
    X_field = scipy.ndimage.zoom(np.ones(y.shape), step)
    Y_field = scipy.ndimage.zoom(np.sin(np.rad2deg(2 * np.pi * x)), step)
    return np.array([X_field, Y_field])


def equation(indices: np.ndarray, step: int = 16) -> np.ndarray:
    x, y = indices.astype(np.float64)
    x, y = x[::step, ::step], y[::step, ::step]
    X_field = scipy.ndimage.zoom(-y, step)
    Y_field = scipy.ndimage.zoom(x, step)
    return np.array([X_field, Y_field])

def render_gif(image_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    images = []
    for image in sorted(Path(image_path).iterdir(), key=os.path.getmtime):
        images.append(imageio.v3.imread(image))
    return imageio.mimsave(output_path, images)


def reset_storage(storage_path: Union[str, Path]) -> None:
    """All this does is delete the given folder and recreate it."""
    shutil.rmtree(storage_path)
    return os.mkdir(storage_path)


def archive_storage(origin_path, archive_path, archive_limit: int = 10):
    """Archives rendered files if necessary."""
    if len(os.listdir(origin_path)) >= archive_limit:
        logger.info(f"Archiving {len(os.listdir(origin_path))} files in {origin_path}.")
        shutil.make_archive(archive_path, "zip", origin_path)
        reset_storage(origin_path)
    return archive_path


def main(frames: int = 100, timestep: float = 1 / 240):
    fluid = FluidDynamics(inflow_dye)
    fluid.velocity_field = equation(fluid.coordinates, 1)

    fluid.render_fluid("./examples/initial_position.png")
    fluid.build_plot("./examples/initial_field.png", grid_step=4)
    logger.debug(f"Velocity Map:\n{fluid.velocity_field}")
    for iteration in range(frames):
        logger.info(f"\x1b[0;33mCurrently rendering frame number {iteration + 1} of {frames}.\x1b[0;0m")
        # It's recommended to not have a timestamp that is greater than 1 / 120.
        fluid.step(timestep=timestep)
        # fluid.build_plot(f"./examples/velocity_storage/velocity_{iteration}.png", grid_step=4)
        fluid.render_fluid(f"./examples/advection_storage/fluid_{iteration}.png")
        logger.info(f"Seconds elapsed: {time.time() - start:.2f} seconds.")

    fluid_output = f"./examples/rendered_fluids/output_{int(time.time())}.gif"
    # velocity_output = f"./examples/rendered_velocities/velocity_{int(time.time())}.gif"
    render_gif("./examples/advection_storage", fluid_output)
    # render_gif("./examples/velocity_storage", velocity_output)


if __name__ == "__main__":
    start = time.time()
    logger.info(f"Rendering fluid with shape {inflow_dye.shape}.")
    # The GPU is not being used by Python.
    reset_storage("./examples/advection_storage")
    reset_storage("./examples/velocity_storage")

    archive_storage(
        origin_path="./examples/rendered_fluids",
        archive_path=f"./examples/archive/archived_fluids/archived_fluids_{int(time.time())}"
    )
    archive_storage(
        origin_path="./examples/rendered_velocities",
        archive_path=f"./examples/archive/archived_velocities/archived_velocities_{int(time.time())}",
    )
    main(frames=50, timestep=1 / 240)
    logger.info(f"Fluid rendered in {time.time() - start:.2f} seconds.")
