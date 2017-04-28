## CS775 - Brittle Fracture Simulation

A submission by Kalpesh Krishna (140070017) and Siddhant Ranade (130260002).

## Requirements

* [VTK](http://www.vtk.org/Wiki/VTK/Examples/Python) - Install this using `sudo apt-get install python-vtk`.
* [Paraview](http://www.paraview.org/download/) - Download the default package in this link.
* [POV-Ray](http://www.povray.org/) - Install this using `sudo apt-get install povray`.

Additionally, to convert STL files to compatible VTK files, you need to use [TetGen](http://wias-berlin.de/software/tetgen/). Download and install it from source (standard `cmake` procedure). It can create VTK files using `./tetgen -kV bunny.stl`. Here are the compatible input [file formats](https://www.wias-berlin.de/software/tetgen/fformats.html).

## Running The Code

To run the simulation script, run `src/simulate.py`. This will create 100 frames in the `output` directory. To view this in Paraview, run `./paraview --data=cube..vtk`. Press the *Play* button to run the simulation.

In the `simulate.py` file, certain parameters can be changed to produce a different output. Using a `body.deform()` function, one can stretch / twist / squash an object. The list of available macros are present in `src/utils/deformations.py`. The `constants` dictionary can be edited to account for other objects and materials.

`namespace` denotes the output file names. `rule` denotes the integration scheme, can be Runge-Kutta4 or Explicit Euler. Finally `ext` points to a set of external forces defined in `src/utils/external.py`.
