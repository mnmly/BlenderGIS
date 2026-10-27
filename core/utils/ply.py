# -*- coding:utf-8 -*-

# This file is part of BlenderGIS

#  ***** GPL LICENSE BLOCK *****
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  All rights reserved.
#  ***** GPL LICENSE BLOCK *****

'''
Write point clouds as binary little endian PLY files, one vertex element with one property per field.
Numpy only.
'''

import re

import numpy as np

#numpy dtype -> PLY property type
PLY_TYPES = {
	np.dtype('int8'): 'char', np.dtype('uint8'): 'uchar',
	np.dtype('int16'): 'short', np.dtype('uint16'): 'ushort',
	np.dtype('int32'): 'int', np.dtype('uint32'): 'uint',
	np.dtype('float32'): 'float', np.dtype('float64'): 'double',
}


def propertyName(name):
	'''PLY property names are single words'''
	return re.sub(r'\W', '_', name.strip()) or '_'

def writePly(path, fields, comments=()):
	'''
	Write the points of fields, an ordered dict {property name: 1d array}, all arrays of the same length
	and of a PLY type (bool arrays are written as uchar). Comments are header lines (no new lines).
	'''
	names, arrays = [], []
	for name, values in fields.items():
		values = np.asarray(values)
		if values.dtype == np.bool_:
			values = values.astype(np.uint8)
		if values.dtype not in PLY_TYPES:
			raise ValueError(f"No PLY type for {name} ({values.dtype})")
		names.append(propertyName(name))
		arrays.append(values)
	if len(set(names)) != len(names):
		raise ValueError(f"Duplicated PLY property names: {names}")
	count = len(arrays[0]) if arrays else 0
	if any(len(values) != count for values in arrays):
		raise ValueError("All PLY properties must have the same number of points")

	dtype = np.dtype([(name, values.dtype.newbyteorder('<')) for name, values in zip(names, arrays)])
	data = np.empty(count, dtype=dtype)
	for name, values in zip(names, arrays):
		data[name] = values

	header = ['ply', 'format binary_little_endian 1.0']
	header += ['comment ' + comment.replace('\n', ' ') for comment in comments]
	header.append(f'element vertex {count}')
	header += [f'property {PLY_TYPES[values.dtype]} {name}' for name, values in zip(names, arrays)]
	header.append('end_header')
	with open(path, 'wb') as f:
		f.write(('\n'.join(header) + '\n').encode('ascii'))
		data.tofile(f)
	return count
