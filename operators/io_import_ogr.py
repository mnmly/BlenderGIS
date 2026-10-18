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

'''Import vector layers of any GDAL/OGR format (GeoPackage, KML, MapInfo, FlatGeobuf, DXF, CSV...) through pyogrio'''

import os
import math
import logging
log = logging.getLogger(__name__)

import bpy
import numpy as np
import pyogrio #bundled as wheel, embeds GDAL (see blender_manifest.toml)
import pyproj
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator

from ..core.utils.wkb import wkbToGeojson, WKBError
from .io_import_geojson import FeaturesImporter, GEOM_CATEGORY

#QGIS 'geometrytype=' datasource option -> GeoJSON geometry category
QGIS_GEOMETRY_CATEGORY = {'Point': 'Point', 'LineString': 'PolyLine', 'Polygon': 'Polygon'}


def toPython(value):
	'''Attribute value read by pyogrio as a value storable in a Blender custom property'''
	if isinstance(value, np.generic):
		value = value.item()
	if isinstance(value, float) and math.isnan(value):
		return None
	if isinstance(value, bytes):
		return None
	if value is not None and not isinstance(value, (bool, int, float, str)):
		return str(value) #dates, times...
	return value

def crsToString(crs):
	'''pyogrio layer crs (AUTH:CODE or WKT) as a string accepted by BlenderGIS (AUTH:CODE or proj4)'''
	if not crs:
		return None
	try:
		crs = pyproj.CRS.from_user_input(crs)
	except pyproj.exceptions.CRSError:
		return None
	authority = crs.to_authority()
	return ':'.join(authority) if authority else crs.to_proj4()

def readOgrFeatures(path, layer=None, where=None, geometryType=None):
	'''
	Read a vector layer, return (layerName, [(geojson geometry dict, properties)], crs string or None)
	geometryType: keep only 'Point', 'LineString' or 'Polygon' features (QGIS splits mixed layers this way)
	'''
	meta, fids, geometries, fieldData = pyogrio.raw.read(path, layer=layer, where=where or None)
	fields = list(meta['fields'])
	category = QGIS_GEOMETRY_CATEGORY.get(geometryType) if geometryType else None
	features = []
	nbSkipped = 0
	for i, data in enumerate(geometries):
		try:
			geom = wkbToGeojson(data)
		except WKBError as e:
			nbSkipped += 1
			log.debug(f"Skipping feature {i}: {e}")
			continue
		if geom is None:
			continue
		props = {name: toPython(values[i]) for name, values in zip(fields, fieldData)}
		parts = geom['geometries'] if geom['type'] == 'GeometryCollection' else [geom]
		for part in parts:
			if category is None or GEOM_CATEGORY.get(part['type']) == category:
				features.append((part, props))
	if nbSkipped:
		log.warning(f"{nbSkipped} features with unsupported geometries (curves) skipped in {path}")
	layerName = meta.get('layer_name') or (layer if isinstance(layer, str) else None) or os.path.splitext(os.path.basename(path))[0]
	return layerName, features, crsToString(meta.get('crs'))


class IMPORTGIS_OT_ogr(FeaturesImporter, Operator):
	"""Import a vector layer of a GDAL/OGR format (GeoPackage, KML, MapInfo, FlatGeobuf, DXF...)"""

	bl_idname = "importgis.ogr"
	bl_description = 'Import a vector layer read by GDAL (GeoPackage, KML, MapInfo, FlatGeobuf, DXF...)'
	bl_label = "Import vector layer (GDAL)"
	bl_options = {"UNDO"}

	layer: StringProperty(name="Layer", description="Layer name (or index) in the file, empty for the first one")
	where: StringProperty(name="Filter", description="SQL attribute filter (WHERE clause)")
	geometryType: StringProperty(name="Geometry type", description="Import only Point, LineString or Polygon features")

	def readFeatures(self):
		layer = int(self.layer) if self.layer.isdigit() else (self.layer or None)
		layerName, features, crs = readOgrFeatures(self.filepath, layer, self.where, self.geometryType)
		return layerName, features


class IMPORTGIS_OT_ogr_file_dialog(Operator, ImportHelper):
	"""Import a vector layer read by GDAL (GeoPackage, KML, MapInfo, FlatGeobuf, DXF...)"""
	bl_idname = "importgis.ogr_file_dialog"
	bl_description = 'Import a vector layer read by GDAL (GeoPackage, KML, MapInfo, FlatGeobuf, DXF...)'
	bl_label = "Import vector (GDAL)"
	bl_options = {'INTERNAL'}

	filter_glob: StringProperty(default="*.gpkg;*.kml;*.kmz;*.gml;*.tab;*.mif;*.fgb;*.dxf;*.csv;*.gpx;*.sqlite;*.geojsonl;*.shp;*.geojson;*.json", options={'HIDDEN'})

	@classmethod
	def poll(cls, context):
		return context.mode == 'OBJECT'

	def execute(self, context):
		try:
			layers = pyogrio.list_layers(self.filepath)
		except Exception as e:
			log.error("Unable to open vector file", exc_info=True)
			self.report({'ERROR'}, f"Unable to open {os.path.basename(self.filepath)}: {e}")
			return {'CANCELLED'}
		if len(layers) == 0:
			self.report({'ERROR'}, "No layer in this file")
			return {'CANCELLED'}
		bpy.ops.importgis.ogr_props_dialog('INVOKE_DEFAULT', filepath=self.filepath)
		return {'FINISHED'}


#Blender needs the strings of dynamic enum items to stay referenced
_layerItems = []
_fieldItems = []

class IMPORTGIS_OT_ogr_props_dialog(Operator):
	bl_idname = "importgis.ogr_props_dialog"
	bl_label = "Import vector (GDAL)"
	bl_options = {'INTERNAL'}

	filepath: StringProperty(options={'HIDDEN'})

	def listLayers(self, context):
		_layerItems[:] = [(str(name), f"{name} ({geomType})", "") for name, geomType in pyogrio.list_layers(self.filepath)]
		return _layerItems

	def updateLayer(self, context):
		try:
			self.crs = crsToString(pyogrio.read_info(self.filepath, layer=self.layer).get('crs')) or ''
		except Exception:
			log.warning("Unable to read layer CRS", exc_info=True)

	layer: EnumProperty(name="Layer", items=listLayers, update=updateLayer)
	crs: StringProperty(name="CRS", description="Coordinate Reference System of the layer (AUTH:CODE or proj4)")

	def listNumericFields(self, context):
		items = [('NONE', 'None', "")]
		try:
			info = pyogrio.read_info(self.filepath, layer=self.layer)
			items += [(name, name, "") for name, dtype in zip(info['fields'], info['dtypes']) if np.issubdtype(np.dtype(dtype), np.number)]
		except Exception:
			log.warning("Unable to read layer fields", exc_info=True)
		_fieldItems[:] = items
		return _fieldItems

	extrusionField: EnumProperty(name="Extrusion field", description="Numeric field used to extrude the features", items=listNumericFields)
	separateObjects: BoolProperty(name="Separate objects", description="One object per feature, with its attributes as custom properties", default=False)

	def invoke(self, context, event):
		self.updateLayer(context)
		return context.window_manager.invoke_props_dialog(self)

	def draw(self, context):
		layout = self.layout
		layout.prop(self, 'layer')
		layout.prop(self, 'crs')
		layout.prop(self, 'extrusionField')
		layout.prop(self, 'separateObjects')

	def execute(self, context):
		if not self.crs:
			self.report({'ERROR'}, "The layer has no CRS, please define one")
			return {'CANCELLED'}
		extrude = '' if self.extrusionField == 'NONE' else self.extrusionField
		return bpy.ops.importgis.ogr(filepath=self.filepath, layer=self.layer, shpCRS=self.crs,
			fieldExtrudeName=extrude, separateObjects=self.separateObjects)


classes = [IMPORTGIS_OT_ogr, IMPORTGIS_OT_ogr_file_dialog, IMPORTGIS_OT_ogr_props_dialog]

def register():
	for cls in classes:
		try:
			bpy.utils.register_class(cls)
		except ValueError:
			log.warning('{} is already registered, now unregister and retry... '.format(cls))
			bpy.utils.unregister_class(cls)
			bpy.utils.register_class(cls)

def unregister():
	for cls in reversed(classes):
		bpy.utils.unregister_class(cls)
