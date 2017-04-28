## Requirements

* [VTK](http://www.vtk.org/Wiki/VTK/Examples/Python) - Install this using `sudo apt-get install python-vtk`.
* [Paraview](http://www.paraview.org/download/) - Download the default package in this link.

Additionally, to convert STL files to compatible VTK files, you need to use [TetGen](http://wias-berlin.de/software/tetgen/). Download and install it from source (standard `cmake` procedure). It can create VTK files using `./tetgen -kV bunny.stl`. Here are the compatible input [file formats](https://www.wias-berlin.de/software/tetgen/fformats.html).

## Running The Code

To run the simulation script, run `src/simulate.py`. This will create 100 frames in the `output` directory. To view this in Paraview, run `./paraview --data=cube..vtk`. Press the *Play* button to run the simulation.
