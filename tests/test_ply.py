'''Unit tests of the PLY writer: uv run --with pytest --with numpy pytest tests/'''

import importlib.util
import os

import pytest

np = pytest.importorskip('numpy')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('ply', os.path.join(ROOT, 'core', 'utils', 'ply.py'))
ply = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ply)

NUMPY_TYPES = {'char': 'i1', 'uchar': 'u1', 'short': '<i2', 'ushort': '<u2', 'int': '<i4', 'uint': '<u4', 'float': '<f4', 'double': '<f8'}


def readPly(path):
	'''Minimal reader of binary little endian PLY vertex elements: (header lines, structured array)'''
	with open(path, 'rb') as f:
		header = []
		while True:
			line = f.readline().decode('ascii').rstrip('\n')
			header.append(line)
			if line == 'end_header':
				break
		count = next(int(line.split()[2]) for line in header if line.startswith('element vertex'))
		dtype = [(line.split()[2], NUMPY_TYPES[line.split()[1]]) for line in header if line.startswith('property')]
		return header, np.fromfile(f, dtype=dtype, count=count)


def test_roundtrip_types_and_header(tmp_path):
	path = str(tmp_path / 'points.ply')
	fields = {
		'x': np.array([525000.123456789, 525999.987654321]), #georeferenced doubles keep their precision
		'y': np.array([180000.5, 180999.25]),
		'z': np.array([12.5, -3.25], dtype=np.float32),
		'red': np.array([255, 0], dtype=np.uint8),
		'intensity': np.array([7, 70000], dtype=np.int32),
		'classification': np.array([2, 6], dtype=np.int8),
		'withheld': np.array([True, False]),
	}
	assert ply.writePly(path, fields, ['crs EPSG:27700']) == 2
	header, data = readPly(path)
	assert header[:3] == ['ply', 'format binary_little_endian 1.0', 'comment crs EPSG:27700']
	assert 'property double x' in header and 'property float z' in header
	assert 'property char classification' in header and 'property uchar withheld' in header
	assert data['x'].tolist() == fields['x'].tolist()
	assert data['intensity'].tolist() == [7, 70000]
	assert data['withheld'].tolist() == [1, 0]

def test_property_names_are_single_words(tmp_path):
	path = str(tmp_path / 'points.ply')
	ply.writePly(path, {'x': np.zeros(1, np.float32), 'scan angle': np.zeros(1, np.float32)})
	assert 'property float scan_angle' in readPly(path)[0]

def test_empty_cloud(tmp_path):
	path = str(tmp_path / 'points.ply')
	assert ply.writePly(path, {'x': np.zeros(0, np.float32)}) == 0
	header, data = readPly(path)
	assert 'element vertex 0' in header and len(data) == 0

@pytest.mark.parametrize('fields', [
	{'x': np.zeros(2, np.float32), 'y': np.zeros(3, np.float32)}, #lengths differ
	{'x': np.zeros(2, np.int64)}, #no PLY type
	{'a b': np.zeros(1, np.float32), 'a_b': np.zeros(1, np.float32)}, #same property name
])
def test_invalid_fields(tmp_path, fields):
	with pytest.raises(ValueError):
		ply.writePly(str(tmp_path / 'points.ply'), fields)
