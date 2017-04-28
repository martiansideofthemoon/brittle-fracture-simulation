"""This class writes POV-Ray files."""
# import numpy as np


class POVRayInterface(object):
	"""docstring for POV-RayInterface."""

	@classmethod
	def write(cls, points, cells, filename):
		"""Write a POV-Ray .pov file."""
		output = \
			'#include "colors.inc"\n' \
			'#include "stones.inc"\n' \
			'#include "textures.inc"\n' \
			'#include "shapes.inc"\n' \
			'#include "glass.inc"\n' \
			'#include "metals.inc"\n' \
			'#include "woods.inc"\n' \
			'background{ White }\n' \
			'camera {\n' \
			'\tangle 15\n' \
			'\tlocation <10, 10, 10>\n' \
			'\tlook_at <0, 0, 0>\n' \
			'}\n' \
			'light_source {\n' \
			'\t<20, 20, -20> color White\n' \
			'}\n' \

		output += \
			'mesh2 {{\n' \
			'\tvertex_vectors {{\n' \
			'\t\t{},\n'.format(points.shape[0])

		for ii in xrange(points.shape[0]):
			output += "\t\t<{}, {}, {}>,\n".format(points[ii, 0], points[ii, 1], points[ii, 2])

		output += '\t}\n'
		output += '\tface_indices {\n'
		output += '\t\t{},\n'.format(cells.shape[0] * 4)

		for ii in xrange(cells.shape[0]):
			output += "\t\t<{}, {}, {}>,\n".format(cells[ii, 0], cells[ii, 1], cells[ii, 2])
			output += "\t\t<{}, {}, {}>,\n".format(cells[ii, 0], cells[ii, 1], cells[ii, 3])
			output += "\t\t<{}, {}, {}>,\n".format(cells[ii, 0], cells[ii, 2], cells[ii, 3])
			output += "\t\t<{}, {}, {}>,\n".format(cells[ii, 1], cells[ii, 2], cells[ii, 3])

		output += '\t}\n'
		output += '\ttexture {DMFWood6}\n'
		output += '}\n'

		with open(filename, 'w') as f:
			f.write(output)
