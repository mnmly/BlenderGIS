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
Read QGIS project files (.qgs xml, or .qgz zip archive containing the .qgs)
and describe their layers so they can be imported with the matching BlenderGIS importer.
Standard library only, no Blender dependency.
'''

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import unquote


#Layer kinds, the importer to use for each layer
SHP = 'SHP'
GEOJSON = 'GEOJSON'
RASTER = 'RASTER'
ASC = 'ASC'
POINTCLOUD = 'POINTCLOUD'
XYZ = 'XYZ'
GPKG = 'GPKG'
UNSUPPORTED = 'UNSUPPORTED'

SUPPORTED_KINDS = {SHP, GEOJSON, RASTER, ASC, POINTCLOUD}

RASTER_EXTENSIONS = {'.tif', '.tiff', '.geotiff', '.jp2', '.png', '.jpg', '.jpeg', '.bmp'}
POINTCLOUD_EXTENSIONS = {'.las', '.laz'}
LOCAL_PROVIDERS = {'ogr', 'gdal', 'pdal', 'copc'}


class QgisProjectError(Exception):
	pass


class Layer():
	'''A layer of the QGIS layer tree (a layer used twice in the tree appears twice)'''

	def __init__(self, id, name):
		self.id = id
		self.name = name
		self.type = None #vector, raster, point-cloud...
		self.provider = None
		self.datasource = ''
		self.path = None #resolved absolute file path for local file layers
		self.subset = None #part of the datasource after '|', e.g. layername=roads
		self.crs = None #authid, e.g. EPSG:2154
		self.crsName = None
		self.geometry = None #Point, Line, Polygon for vector layers
		self.extent = None #(xmin, ymin, xmax, ymax) in layer crs
		self.visible = False #checked, with all its parent groups checked
		self.groupPath = () #names of the parent groups, from the root
		self.order = 0 #position in the layer tree, 0 is the top (drawn above the others)
		self.color = None #(r, g, b, a) floats of a single symbol renderer
		self.kind = UNSUPPORTED
		self.reason = None #why the layer cannot be imported

	@property
	def supported(self):
		return self.kind in SUPPORTED_KINDS and self.reason is None

	def __repr__(self):
		return f"Layer({self.name!r}, kind={self.kind}, crs={self.crs}, visible={self.visible}, reason={self.reason!r})"


class QgisProject():

	def __init__(self, path, root):
		self.path = path
		self.name = os.path.splitext(os.path.basename(path))[0]
		self.directory = os.path.dirname(os.path.abspath(path))
		self.version = root.get('version')
		self.crs = root.findtext('./projectCrs/spatialrefsys/authid') or None
		self.crsName = root.findtext('./projectCrs/spatialrefsys/description') or None
		self.canvasExtent = _readExtent(root.find('.//mapcanvas/extent'))
		self._mapLayers = {ml.findtext('id'): ml for ml in root.findall('./projectlayers/maplayer')}
		self.layers = []
		treeRoot = root.find('layer-tree-group')
		if treeRoot is not None:
			self._walkTree(treeRoot, (), True)

	@classmethod
	def load(cls, path):
		'''Read a .qgs or .qgz project file'''
		try:
			if zipfile.is_zipfile(path):
				with zipfile.ZipFile(path) as z:
					qgsNames = [n for n in z.namelist() if n.lower().endswith('.qgs')]
					if not qgsNames:
						raise QgisProjectError("No .qgs project inside " + path)
					content = z.read(qgsNames[0])
			else:
				with open(path, 'rb') as f:
					content = f.read()
			root = ET.fromstring(content)
		except (OSError, zipfile.BadZipFile, ET.ParseError) as e:
			raise QgisProjectError(f"Cannot read QGIS project {path}: {e}") from e
		if root.tag != 'qgis':
			raise QgisProjectError(f"{path} is not a QGIS project")
		return cls(path, root)

	@property
	def canvasCenter(self):
		if self.canvasExtent is None:
			return None
		xmin, ymin, xmax, ymax = self.canvasExtent
		return (xmin + xmax) / 2, (ymin + ymax) / 2

	def _walkTree(self, group, groupPath, parentVisible):
		for node in group:
			if node.tag == 'layer-tree-group':
				visible = parentVisible and _isChecked(node.get('checked'))
				self._walkTree(node, groupPath + (node.get('name', ''),), visible)
			elif node.tag == 'layer-tree-layer':
				layer = Layer(node.get('id'), node.get('name'))
				layer.groupPath = groupPath
				layer.visible = parentVisible and _isChecked(node.get('checked'))
				layer.order = len(self.layers)
				mapLayer = self._mapLayers.get(layer.id)
				if mapLayer is None:
					layer.reason = "Layer definition missing from the project"
				else:
					self._readMapLayer(layer, mapLayer)
				self.layers.append(layer)

	def _readMapLayer(self, layer, ml):
		layer.type = ml.get('type')
		layer.provider = ml.findtext('provider')
		layer.datasource = ml.findtext('datasource') or ''
		layer.name = layer.name or ml.findtext('layername')
		layer.crs = ml.findtext('./srs/spatialrefsys/authid') or None
		layer.crsName = ml.findtext('./srs/spatialrefsys/description') or None
		layer.geometry = ml.get('geometry')
		layer.extent = _readExtent(ml.find('extent'))
		layer.color = _readSymbolColor(ml)
		if layer.provider in LOCAL_PROVIDERS:
			path, layer.subset = self._resolvePath(layer.datasource)
			layer.path = path
		layer.kind, layer.reason = _classify(layer)
		if layer.reason is None and layer.path is not None and not os.path.exists(layer.path):
			layer.reason = "File not found: " + layer.path
		if layer.reason is None and layer.kind in SUPPORTED_KINDS and not layer.crs:
			layer.reason = "Layer has no CRS in the project"

	def _resolvePath(self, datasource):
		'''Absolute file path of a local datasource, and its optional '|' suffix (e.g. layername=roads)'''
		path, _, subset = datasource.partition('|')
		if path.startswith('file://'):
			path = unquote(path[len('file://'):])
		if not (os.path.isabs(path) or re.match(r'^[A-Za-z]:[\\/]', path)):
			path = os.path.join(self.directory, path)
		return os.path.normpath(path), (subset or None)


def _isChecked(value):
	return value in ('Qt::Checked', '2', '1', 'true', 'True')

def _readExtent(node):
	if node is None:
		return None
	try:
		return tuple(float(node.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
	except (TypeError, ValueError):
		return None

def _parseColor(value):
	'''QGIS color option "r,g,b,a" (QGIS 3.40+ may append ",rgb:..."), to floats in [0, 1]'''
	parts = value.split(',')
	try:
		rgba = [int(p) for p in parts[:4]]
	except ValueError:
		return None
	if len(rgba) == 3:
		rgba.append(255)
	return tuple(c / 255 for c in rgba) if len(rgba) == 4 else None

def _readSymbolColor(ml):
	'''Color of a single symbol vector renderer (fill color for polygons, line color for lines)'''
	renderer = ml.find('renderer-v2')
	if renderer is None or renderer.get('type') != 'singleSymbol':
		return None
	symbolLayer = renderer.find('./symbols/symbol/layer')
	if symbolLayer is None:
		return None
	options = {o.get('name'): o.get('value') for o in symbolLayer.iter('Option') if o.get('name')}
	options.update({p.get('k'): p.get('v') for p in symbolLayer.iter('prop')}) #QGIS < 3.26 style
	for key in ('color', 'line_color', 'outline_color'):
		if options.get(key):
			color = _parseColor(options[key])
			if color is not None:
				return color
	return None

def _classify(layer):
	'''Return (kind, reason) for a layer, reason is None when it can be imported'''
	ext = os.path.splitext(layer.path)[1].lower() if layer.path else ''
	if layer.type == 'vector':
		if layer.provider != 'ogr':
			return UNSUPPORTED, f"Vector provider '{layer.provider}' is not supported"
		if ext == '.shp':
			return SHP, None
		if ext in ('.geojson', '.json'):
			return GEOJSON, None
		if ext == '.gpkg':
			return GPKG, "GeoPackage layers are not supported yet"
		return UNSUPPORTED, f"Vector format '{ext or layer.datasource[:40]}' is not supported yet"
	if layer.type == 'raster':
		if layer.provider == 'wms':
			if 'type=xyz' in layer.datasource:
				return XYZ, "Web basemaps (XYZ tiles) are not supported yet"
			return UNSUPPORTED, "WMS/WMTS layers are not supported"
		if layer.provider != 'gdal':
			return UNSUPPORTED, f"Raster provider '{layer.provider}' is not supported"
		if ext in RASTER_EXTENSIONS:
			return RASTER, None
		if ext == '.asc':
			return ASC, None
		return UNSUPPORTED, f"Raster format '{ext}' is not supported"
	if layer.type == 'point-cloud':
		if layer.provider in ('pdal', 'copc') and ext in POINTCLOUD_EXTENSIONS:
			return POINTCLOUD, None
		return UNSUPPORTED, "Only LAS/LAZ point clouds are supported"
	return UNSUPPORTED, f"Layer type '{layer.type}' is not supported"
