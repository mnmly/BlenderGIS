'''Unit tests of the WKB decoder: uv run --with pytest pytest tests/'''

import importlib.util
import os
import struct

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('wkb', os.path.join(ROOT, 'core', 'utils', 'wkb.py'))
wkb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wkb)


def header(wkbType, little=True):
	return struct.pack('<BI' if little else '>BI', 1 if little else 0, wkbType)

def pts(points, little=True):
	fmt = '<' if little else '>'
	data = struct.pack(fmt + 'I', len(points))
	for p in points:
		data += struct.pack(fmt + 'd' * len(p), *p)
	return data


def test_point_2d_3d_and_big_endian():
	assert wkb.wkbToGeojson(header(1) + struct.pack('<dd', 1.5, 2.5)) == {'type': 'Point', 'coordinates': [1.5, 2.5]}
	assert wkb.wkbToGeojson(header(1001) + struct.pack('<ddd', 1, 2, 3)) == {'type': 'Point', 'coordinates': [1, 2, 3]}
	assert wkb.wkbToGeojson(header(1, little=False) + struct.pack('>dd', 7, 8)) == {'type': 'Point', 'coordinates': [7, 8]}

def test_m_values_are_dropped():
	#ISO PointM (2001), PointZM (3001) and EWKB LineString with Z and M flags
	assert wkb.wkbToGeojson(header(2001) + struct.pack('<ddd', 1, 2, 99)) == {'type': 'Point', 'coordinates': [1, 2]}
	assert wkb.wkbToGeojson(header(3001) + struct.pack('<dddd', 1, 2, 3, 99)) == {'type': 'Point', 'coordinates': [1, 2, 3]}
	ewkb = header(2 | wkb.EWKB_Z | wkb.EWKB_M) + pts([(0, 0, 5, 99), (1, 1, 6, 99)])
	assert wkb.wkbToGeojson(ewkb) == {'type': 'LineString', 'coordinates': [[0, 0, 5], [1, 1, 6]]}

def test_ewkb_srid_is_skipped():
	data = struct.pack('<BII', 1, 1 | wkb.EWKB_SRID, 2154) + struct.pack('<dd', 3, 4)
	assert wkb.wkbToGeojson(data) == {'type': 'Point', 'coordinates': [3, 4]}

def test_polygon_with_hole_and_multipolygon():
	outer = [(0, 0), (10, 0), (10, 10), (0, 0)]
	hole = [(2, 2), (3, 2), (3, 3), (2, 2)]
	poly = header(3) + struct.pack('<I', 2) + pts(outer) + pts(hole)
	expected = [[list(p) for p in outer], [list(p) for p in hole]]
	assert wkb.wkbToGeojson(poly) == {'type': 'Polygon', 'coordinates': expected}
	multi = header(6) + struct.pack('<I', 2) + poly + poly
	assert wkb.wkbToGeojson(multi) == {'type': 'MultiPolygon', 'coordinates': [expected, expected]}

def test_multipoint_multilinestring_and_collection():
	mp = header(4) + struct.pack('<I', 2) + header(1) + struct.pack('<dd', 1, 2) + header(1) + struct.pack('<dd', 3, 4)
	assert wkb.wkbToGeojson(mp) == {'type': 'MultiPoint', 'coordinates': [[1, 2], [3, 4]]}
	ml = header(5) + struct.pack('<I', 1) + header(2) + pts([(0, 0), (1, 1)])
	assert wkb.wkbToGeojson(ml) == {'type': 'MultiLineString', 'coordinates': [[[0, 0], [1, 1]]]}
	gc = header(7) + struct.pack('<I', 2) + header(1) + struct.pack('<dd', 1, 2) + header(2) + pts([(0, 0), (1, 1)])
	assert wkb.wkbToGeojson(gc) == {'type': 'GeometryCollection', 'geometries': [
		{'type': 'Point', 'coordinates': [1, 2]}, {'type': 'LineString', 'coordinates': [[0, 0], [1, 1]]}]}

def test_empty_and_errors():
	assert wkb.wkbToGeojson(None) is None
	assert wkb.wkbToGeojson(b'') is None
	nan = float('nan')
	assert wkb.wkbToGeojson(header(1) + struct.pack('<dd', nan, nan)) == {'type': 'Point', 'coordinates': []}
	with pytest.raises(wkb.WKBError):
		wkb.wkbToGeojson(header(8) + pts([(0, 0), (1, 1), (2, 0)])) #CircularString
	with pytest.raises(wkb.WKBError):
		wkb.wkbToGeojson(header(2) + struct.pack('<I', 5)) #truncated
