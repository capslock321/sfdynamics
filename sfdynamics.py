import os
import sys
import time
import shutil
import logging
import argparse

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

parser = argparse.ArgumentParser(
    prog="SFDynamics",
    description="A Python implementation of Stable Fluids",
)

parser.add_argument("frames", help="The amount of frames to generate", type=int, nargs="?", default=100)
parser.add_argument("timestep", help="The timestep, or how fast to simulate", type=float, nargs="?", default=1/240)
parser.add_argument("resolution", help="The resolution of the simulation.", type=int, nargs="?", default=8)

parser.add_argument("-z", "--zoom", default=16, dest="zoom", help="Zooms the initial inflow array by the given amount.")
parser.add_argument("-t", "--temp-path", default=".temp", dest="temp", help="The location where the rendering frames are stored.")
parser.add_argument("-o", "--output", default="output", dest="output", help="The file path to the final generated GIF.")

arguments = parser.parse_args()

RESOLUTION = (arguments.resolution,) * 2
# Creates a checkerboard pattern
inflow_dye = np.indices(RESOLUTION).sum(axis=0) % 2
inflow_dye = np.kron(inflow_dye, np.ones((arguments.zoom,) * 2))
inflow_dye = inflow_dye.astype(np.uint8)

def generate_initial_velocity(indices: np.ndarray, step: int = 16) -> np.ndarray:
    x, y = indices.astype(np.float64)
    x, y = x[::step, ::step], y[::step, ::step]
    X_field = scipy.ndimage.zoom(2*np.pi*y, step)
    Y_field = scipy.ndimage.zoom(2*np.pi*x, step)
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


def main(frames: int = 100, timestep: float = 1 / 240) -> None:
    fluid = FluidDynamics(inflow_dye)
    fluid.velocity_field = generate_initial_velocity(fluid.coordinates, 1)

    for iteration in range(frames):
        logger.info(f"\x1b[0;33mCurrently rendering frame number {iteration + 1} of {frames}.\x1b[0;0m")
        # It's recommended to not have a timestep that is greater than 1 / 120.
        fluid.step(timestep=timestep)
        fluid.render_fluid(f"{arguments.temp}\\fluid_{iteration}.png")
        logger.info(f"Seconds elapsed: {time.time() - start:.2f} seconds.")

    fluid_output = f"{arguments.output}\\output_{int(time.time())}.gif"
    return render_gif(arguments.temp, fluid_output) # Returns None


if __name__ == "__main__":
    start = time.time()
    logger.info(f"Rendering fluid with shape {inflow_dye.shape}.")
    # The GPU is not being used by Python, meaning that rendering is slow/nonexistent.
    reset_storage(arguments.temp) # clears the temporary cache before starting generation
    main(frames=arguments.frames, timestep=arguments.timestep)
    logger.info(f"Fluid rendered in {time.time() - start:.2f} seconds.")
