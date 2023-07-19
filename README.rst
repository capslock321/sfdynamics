sfdynamics
==================================
A Stable Fluids implementation in Python using only Numpy and PIL.

Installation and Execution
----------------------------------
At it's core, this project currently uses only three dependencies: Matplotlib, Pillow, and Numpy. However, for
a quick and easy setup such as GIF generation, additional dependencies (imageio, scipy) must be installed.

.. code-block:: bash

    python -m pip install -r requirements.txt

Afterwards you can just run `sfdynamics.py` and *hopefully* your fluid GIF should begin generating.

.. code-block:: bash

    python sfdynamics.py

Sources
----------------------------------
Throughout the course of this project, I used many sources with these being the most noteworthy.

- Benedikt Bitterli's Incremental Fluids: https://github.com/tunabrain/incremental-fluids
- GregTJ's Stable Fluids: https://github.com/GregTJ/stable-fluids
- Jamie Wong's interactive implementation: https://jamie-wong.com/2016/08/05/webgl-fluid-simulation/

Copyright and Licence
----------------------------------
This project is licenced under the MIT licence.
