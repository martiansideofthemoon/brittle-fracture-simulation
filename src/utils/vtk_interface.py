"""This class can read and write VTK files."""
import numpy as np
import vtk


class VTKInterface(object):
    """This class reads a vtk file and returns numpy data."""

    @classmethod
    def read(cls, filename):
        """The function reads a given VTK file."""
        # Read the source file.
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        output = reader.GetOutput()
        points = np.zeros((output.GetNumberOfPoints(), 3))
        for i in range(len(points)):
            points[i] = np.array(output.GetPoint(i))
        cells = np.zeros((output.GetNumberOfCells(), 4))
        for i in range(len(cells)):
            cells[i][0] = output.GetCell(i).GetPointId(0)
            cells[i][1] = output.GetCell(i).GetPointId(1)
            cells[i][2] = output.GetCell(i).GetPointId(2)
            cells[i][3] = output.GetCell(i).GetPointId(3)
        return points, cells

    @classmethod
    def write(cls, points, cells, filename):
        """Dump points to an unstructured grid VTK file."""
        output = \
            "# vtk DataFile Version 2.0\n" + \
            "Unstructured Grid\n" + \
            "ASCII\n" + \
            "DATASET UNSTRUCTURED_GRID\n"

        output += "POINTS " + str(len(points)) + " double\n"
        for point in points:
            output += str(point[0]) + " " + str(point[1]) + " " + str(point[2])
            output += "\n"
        output += "\nCELLS " + str(len(cells)) + " " + str(5 * len(cells))
        output += "\n"
        cell_types = ""
        for cell in cells:
            output += "4  " + \
                str(int(cell[0])) + " " + \
                str(int(cell[1])) + " " + \
                str(int(cell[2])) + " " + \
                str(int(cell[3])) + "\n"
            cell_types += "10\n"
        output += "\nCELL_TYPES " + str(len(cells)) + "\n" + cell_types
        with open(filename, 'w') as f:
            f.write(output)
