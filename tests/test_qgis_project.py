'''
Unit tests of the QGIS project reader, no Blender needed:
	uv run --with pytest pytest tests/
'''

import importlib.util
import os
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#load the module by path: importing the core package pulls Blender related modules
_spec = importlib.util.spec_from_file_location('qgis_project', os.path.join(ROOT, 'core', 'qgis', 'project.py'))
qp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qp)


def maplayer(id, type, provider, datasource, crs='EPSG:2154', geometry=None, renderer=''):
	geom = f' geometry="{geometry}"' if geometry else ''
	return f'''
	<maplayer type="{type}"{geom}>
		<extent><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>20</ymax></extent>
		<id>{id}</id>
		<datasource>{datasource}</datasource>
		<layername>{id}</layername>
		<srs><spatialrefsys><authid>{crs}</authid><description>Some CRS</description></spatialrefsys></srs>
		<provider>{provider}</provider>
		{renderer}
	</maplayer>'''

SINGLE_LINE_SYMBOL = '''
	<renderer-v2 type="singleSymbol"><symbols><symbol type="line" name="0"><layer class="SimpleLine">
		<Option type="Map"><Option name="line_color" type="QString" value="141,90,153,255,rgb:0.55,0.35,0.6,1"/></Option>
	</layer></symbol></symbols></renderer-v2>'''

def project(tree, layers, crs='EPSG:2154'):
	return f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.40.5-Bratislava" projectname="">
	<projectCrs><spatialrefsys><authid>{crs}</authid><description>RGF93 v1 / Lambert-93</description></spatialrefsys></projectCrs>
	<layer-tree-group>{tree}</layer-tree-group>
	<mapcanvas name="theMapCanvas"><extent><xmin>100</xmin><ymin>200</ymin><xmax>300</xmax><ymax>600</ymax></extent></mapcanvas>
	<projectlayers>{''.join(layers)}</projectlayers>
</qgis>'''

def treeLayer(id, checked=True):
	return f'<layer-tree-layer id="{id}" name="{id}" checked="{"Qt::Checked" if checked else "Qt::Unchecked"}"/>'

@pytest.fixture
def dataDir(tmp_path):
	d = tmp_path / 'data'
	d.mkdir()
	for name in ('roads.shp', 'pois.geojson', 'ortho.tif', 'dem.asc', 'cloud.laz', 'city.gpkg', 'trace.kml'):
		(d / name).write_bytes(b'')
	return d

def write(tmp_path, xml, name='proj.qgs'):
	path = tmp_path / name
	path.write_text(xml, encoding='utf-8')
	return str(path)


def test_kinds_paths_and_crs(tmp_path, dataDir):
	layers = [
		maplayer('roads', 'vector', 'ogr', './data/roads.shp', geometry='Line', renderer=SINGLE_LINE_SYMBOL),
		maplayer('pois', 'vector', 'ogr', './data/pois.geojson', crs='EPSG:4326', geometry='Point'),
		maplayer('ortho', 'raster', 'gdal', './data/ortho.tif'),
		maplayer('dem', 'raster', 'gdal', './data/dem.asc'),
		maplayer('cloud', 'point-cloud', 'pdal', './data/cloud.laz'),
		maplayer('city', 'vector', 'ogr', './data/city.gpkg|layername=buildings', geometry='Polygon'),
		maplayer('trace', 'vector', 'ogr', './data/trace.kml'),
		maplayer('osm', 'raster', 'wms', 'type=xyz&amp;url=https://tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png', crs='EPSG:3857'),
		maplayer('db', 'vector', 'postgres', "dbname='gis' table=roads"),
	]
	tree = ''.join(treeLayer(i) for i in ('roads', 'pois', 'ortho', 'dem', 'cloud', 'city', 'trace', 'osm', 'db'))
	prj = qp.QgisProject.load(write(tmp_path, project(tree, layers)))
	byId = {l.id: l for l in prj.layers}

	assert prj.crs == 'EPSG:2154' and prj.crsName == 'RGF93 v1 / Lambert-93'
	assert prj.canvasExtent == (100, 200, 300, 600) and prj.canvasCenter == (200, 400)
	assert [byId[i].kind for i in ('roads', 'pois', 'ortho', 'dem', 'cloud')] == [qp.SHP, qp.GEOJSON, qp.RASTER, qp.ASC, qp.POINTCLOUD]
	assert all(byId[i].supported for i in ('roads', 'pois', 'ortho', 'dem', 'cloud'))
	assert byId['roads'].path == str(dataDir / 'roads.shp')
	assert byId['pois'].crs == 'EPSG:4326'
	assert byId['city'].kind == qp.OGR and byId['city'].supported and byId['city'].ogr == {'layer': 'buildings', 'where': None, 'geometryType': None}
	assert byId['trace'].kind == qp.OGR and byId['trace'].supported and byId['trace'].ogr['layer'] is None
	assert byId['osm'].kind == qp.XYZ and byId['osm'].supported and byId['osm'].path is None
	assert byId['osm'].xyz == {'url': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', 'zmin': 0, 'zmax': 19, 'referer': ''}
	assert byId['db'].kind == qp.UNSUPPORTED and 'postgres' in byId['db'].reason
	assert byId['roads'].color == pytest.approx((141 / 255, 90 / 255, 153 / 255, 1))
	assert [l.order for l in prj.layers] == list(range(9))


def test_group_visibility_and_paths(tmp_path, dataDir):
	tree = f'''
		<layer-tree-group name="Base" checked="Qt::Checked">
			{treeLayer('roads')}
			<layer-tree-group name="Hidden" checked="Qt::Unchecked">{treeLayer('ortho')}</layer-tree-group>
		</layer-tree-group>
		{treeLayer('pois', checked=False)}
		{treeLayer('roads')}'''
	layers = [
		maplayer('roads', 'vector', 'ogr', str(dataDir / 'roads.shp')), #absolute path
		maplayer('ortho', 'raster', 'gdal', 'file://' + str(dataDir / 'ortho.tif')),
		maplayer('pois', 'vector', 'ogr', './data/pois.geojson'),
	]
	prj = qp.QgisProject.load(write(tmp_path, project(tree, layers)))
	names = [(l.id, l.groupPath, l.visible) for l in prj.layers]
	assert names == [
		('roads', ('Base',), True),
		('ortho', ('Base', 'Hidden'), False), #checked itself, but its group is not
		('pois', (), False),
		('roads', (), True), #a layer used twice in the tree is listed twice
	]
	assert prj.layers[1].path == str(dataDir / 'ortho.tif')


def test_missing_file_and_missing_definition(tmp_path):
	tree = treeLayer('gone') + treeLayer('ghost')
	layers = [maplayer('gone', 'raster', 'gdal', '../nowhere/ortho.tif')]
	prj = qp.QgisProject.load(write(tmp_path, project(tree, layers)))
	gone, ghost = prj.layers
	assert gone.kind == qp.RASTER and not gone.supported and gone.reason.startswith('File not found')
	assert not ghost.supported and 'missing' in ghost.reason


def test_qgz_archive(tmp_path, dataDir):
	xml = project(treeLayer('roads'), [maplayer('roads', 'vector', 'ogr', './data/roads.shp')])
	qgz = tmp_path / 'proj.qgz'
	with zipfile.ZipFile(qgz, 'w') as z:
		z.writestr('proj.qgs', xml)
		z.writestr('proj_styles.db', b'')
	prj = qp.QgisProject.load(str(qgz))
	assert prj.name == 'proj' and prj.layers[0].supported and prj.layers[0].path == str(dataDir / 'roads.shp')


def test_not_a_project(tmp_path):
	path = tmp_path / 'bad.qgs'
	path.write_text('<html/>')
	with pytest.raises(qp.QgisProjectError):
		qp.QgisProject.load(str(path))
	path.write_text('not xml at all')
	with pytest.raises(qp.QgisProjectError):
		qp.QgisProject.load(str(path))


def test_ogr_options(tmp_path, dataDir):
	layers = [
		maplayer('pts', 'vector', 'ogr', "./data/city.gpkg|layername=pois|subset=\"kind\" = 'shop'|geometrytype=Point"),
		maplayer('byid', 'vector', 'ogr', './data/trace.kml|layerid=2'),
	]
	layers += [
		maplayer('folder', 'vector', 'ogr', './data|layername=roads'), #folder of shapefiles
		maplayer('remote', 'vector', 'ogr', '/vsicurl/https://example.org/roads.fgb'),
	]
	prj = qp.QgisProject.load(write(tmp_path, project(''.join(treeLayer(i) for i in ('pts', 'byid', 'folder', 'remote')), layers)))
	pts, byid, folder, remote = prj.layers
	assert folder.kind == qp.OGR and folder.supported and folder.path == str(dataDir) and folder.ogr['layer'] == 'roads'
	assert remote.kind == qp.UNSUPPORTED and 'Remote' in remote.reason
	assert pts.supported and pts.ogr == {'layer': 'pois', 'where': '"kind" = \'shop\'', 'geometryType': 'Point'}
	assert byid.supported and byid.ogr == {'layer': 2, 'where': None, 'geometryType': None}


def test_xyz_variants(tmp_path):
	layers = [
		maplayer('tms', 'raster', 'wms', 'crs=EPSG:3857&amp;format&amp;type=xyz&amp;url=https://tiles.example.org/%7Bz%7D/%7Bx%7D/%7B-y%7D.jpg&amp;zmax=17&amp;zmin=2&amp;http-header:referer=https://example.org', crs=''),
		maplayer('bad', 'raster', 'wms', 'type=xyz&amp;url=https://tiles.example.org/tile.png', crs='EPSG:3857'),
		maplayer('wmts', 'raster', 'wms', 'contextualWMSLegend=0&amp;layers=ortho&amp;url=https://wmts.example.org', crs='EPSG:3857'),
	]
	prj = qp.QgisProject.load(write(tmp_path, project(''.join(treeLayer(i) for i in ('tms', 'bad', 'wmts')), layers)))
	tms, bad, wmts = prj.layers
	assert tms.supported and tms.xyz == {'url': 'https://tiles.example.org/{z}/{x}/{-y}.jpg', 'zmin': 2, 'zmax': 17, 'referer': 'https://example.org'}
	assert bad.kind == qp.XYZ and not bad.supported and 'template' in bad.reason
	assert wmts.kind == qp.UNSUPPORTED and 'WMS' in wmts.reason


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'demo.qgz')), reason="local demo project not available")
def test_demo_project():
	prj = qp.QgisProject.load(os.path.join(ROOT, 'demo.qgz'))
	assert prj.crs == 'EPSG:31370'
	assert [l.kind for l in prj.layers] == [qp.RASTER, qp.POINTCLOUD, qp.XYZ] #visibility follows the user's edits of the file
	assert prj.layers[1].crs == 'EPSG:31370'


def test_custom_crs_without_authid(tmp_path, dataDir):
	#QGIS stores a CRS read from a LAS header (e.g. a compound CRS) as wkt, with an empty authid
	wkt = 'COMPOUNDCRS["NAD83 / UTM zone 10N + NAVD88 height",...]'
	layer = maplayer('cloud', 'point-cloud', 'pdal', './data/cloud.laz', crs='').replace('<authid></authid>', f'<authid></authid><wkt>{wkt}</wkt>')
	xml = project(treeLayer('cloud'), [layer])
	cloud, = qp.QgisProject.load(write(tmp_path, xml)).layers
	assert not cloud.supported and 'no CRS' in cloud.reason
	seen = []
	resolve = lambda w, p: seen.append(w) or 'EPSG:26910'
	cloud, = qp.QgisProject.load(write(tmp_path, xml), resolve).layers
	assert cloud.supported and cloud.crs == 'EPSG:26910' and seen == [wkt]
