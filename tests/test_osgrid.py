'''Unit tests of the Ordnance Survey grid references: uv run --with pytest pytest tests/'''

import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('osgrid', os.path.join(ROOT, 'core', 'proj', 'osgrid.py'))
osgrid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(osgrid)


@pytest.mark.parametrize('letters, digits, quadrant, extent', [
	('SV', '', '', (0, 0, 100000)), #false origin
	('TQ', '', '', (500000, 100000, 100000)),
	('HP', '', '', (400000, 1200000, 100000)), #Shetland
	('TQ', '28', '', (520000, 180000, 10000)),
	('TQ', '2580', '', (525000, 180000, 1000)),
	('NT', '2473', '', (324000, 673000, 1000)), #Edinburgh
	('TQ', '2580012345', '', (525800, 112345, 1)),
	('TQ', '28', 'SE', (525000, 180000, 5000)),
	('TQ', '28', 'NW', (520000, 185000, 5000)),
	('tq', '28', 'ne', (525000, 185000, 5000)),
])
def test_grid_ref_extent(letters, digits, quadrant, extent):
	assert osgrid.gridRefExtent(letters, digits, quadrant) == extent

@pytest.mark.parametrize('letters, digits', [('TI', ''), ('AA', ''), ('ZZ', ''), ('TQ', '258')])
def test_invalid_grid_refs(letters, digits):
	assert osgrid.gridRefExtent(letters, digits) is None

@pytest.mark.parametrize('name, ref', [
	('TQ2580_P_8301_20120402_20120416', 'TQ2580'),
	('TQ2580_P_8301_20120402_20120416.copc', 'TQ2580'),
	('/data/tq2580.laz', 'TQ2580'),
	('TQ28se_dtm', 'TQ28SE'),
	('SU', 'SU'),
	('TQ25801_P', None), #odd digit count
	('TQ2580ABC', None),
	('Survey_TQ2580', None), #not at the start
	('AA1234_scan', None), #outside the grid
])
def test_grid_ref_from_file_name(name, ref):
	found = osgrid.gridRefFromFileName(name)
	assert (found[0] if found else None) == ref

def test_match_tile_checks_bounds():
	name = 'TQ2580_P_8301_20120402_20120416'
	#header bounds of the Environment Agency tile
	assert osgrid.matchTile(name, (525000.0, 180000.0, -19.96), (526000.0, 181000.0, 305.64)) == 'TQ2580'
	#tiles overlapping their neighbours a little
	assert osgrid.matchTile(name, (524980, 179990, 0), (526020, 181010, 0)) == 'TQ2580'
	#another square of the grid, or another CRS (UTM zone 30N)
	assert osgrid.matchTile(name, (524000, 180000, 0), (525000, 181000, 0)) is None
	assert osgrid.matchTile(name, (699000, 5710000, 0), (700000, 5711000, 0)) is None
	assert osgrid.matchTile('scan_001', (525000, 180000, 0), (526000, 181000, 0)) is None
