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
Export point clouds to PLY. Unlike the Blender PLY exporter, it handles point cloud objects and
BlenderGIS point clouds with their clip and section applied, and writes georeferenced coordinates in 64 bit
'''

import logging
log = logging.getLogger(__name__)

import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator

from ..geoscene import GeoScene
from ..core.utils.ply import writePly
from .pointcloud_display import hasPoints
from .utils.pointcloud_utils import COLOR_ATTRIBUTE, UV_ATTRIBUTE

#attributes that are not point data: added by the display (radius) or internal to BlenderGIS
SKIPPED_ATTRIBUTES = {'position', 'radius', UV_ATTRIBUTE}
#Blender attribute type -> (foreach_get key, numpy dtype)
SCALAR_TYPES = {
	'FLOAT': ('value', np.float32),
	'INT': ('value', np.int32),
	'INT8': ('value', np.int8),
	'BOOLEAN': ('value', np.bool_),
}
COLOR_TYPES = {'FLOAT_COLOR', 'BYTE_COLOR'}


def pointData(obj, depsgraph, applyModifiers):
	'''
	Mesh or point cloud data holding the points of obj, evaluated (clip and section applied) or as stored,
	and its owner: the evaluated geometry set must be kept referenced while the data is read
	'''
	if not applyModifiers:
		return obj.data, obj
	geometry = obj.evaluated_get(depsgraph).evaluated_geometry()
	#BlenderGIS point clouds are meshes turned into a point cloud by their display modifier
	if geometry.pointcloud is not None and len(geometry.pointcloud.points):
		return geometry.pointcloud, geometry
	return geometry.mesh, geometry

def readAttribute(attr, count, key, dtype, width=1):
	values = np.empty(count * width, dtype=dtype)
	attr.data.foreach_get(key, values)
	return values.reshape(count, width) if width > 1 else values

def readPoints(obj, depsgraph, applyModifiers, withAttributes):
	'''World positions (n, 3) float64, sRGB colors (n, 3) uint8 or None, and {name: (n,) array} of the scalar attributes'''
	data, owner = pointData(obj, depsgraph, applyModifiers)
	if data is None:
		return np.empty((0, 3)), None, {}
	attributes = data.attributes
	count = len(attributes['position'].data)
	co = readAttribute(attributes['position'], count, 'vector', np.float32, 3).astype(np.float64)
	matrix = np.array(obj.matrix_world, dtype=np.float64)
	xyz = co @ matrix[:3, :3].T + matrix[:3, 3]

	pointAttributes = [attr for attr in attributes
		if attr.domain == 'POINT' and not attr.name.startswith('.') and attr.name not in SKIPPED_ATTRIBUTES]
	colors = None
	colorAttrs = [attr for attr in pointAttributes if attr.data_type in COLOR_TYPES]
	if colorAttrs:
		colorAttr = next((attr for attr in colorAttrs if attr.name == COLOR_ATTRIBUTE), colorAttrs[0])
		rgba = readAttribute(colorAttr, count, 'color_srgb', np.float32, 4)
		colors = np.round(np.clip(rgba[:, :3], 0, 1) * 255).astype(np.uint8)

	scalars = {}
	if withAttributes:
		for attr in pointAttributes:
			if attr.data_type in SCALAR_TYPES:
				key, dtype = SCALAR_TYPES[attr.data_type]
				scalars[attr.name] = readAttribute(attr, count, key, dtype)
	return xyz, colors, scalars


class EXPORTGIS_OT_ply(Operator, ExportHelper):
	"""Export the selected point clouds to a PLY file"""
	bl_idname = "exportgis.ply"
	bl_description = 'Export the selected point clouds to PLY, with their clip and section applied'
	bl_label = "Export PLY"
	bl_options = {"UNDO"}

	filename_ext = ".ply"
	filter_glob: StringProperty(default="*.ply", options={'HIDDEN'})

	coordinates: EnumProperty(
		name="Coordinates",
		items=[
			('GEO', 'Georeferenced', "Coordinates in the scene CRS, as 64 bit doubles (for CloudCompare, QGIS, PDAL...)"),
			('SCENE', 'Scene', "Blender world coordinates, as 32 bit floats (for 3D software)"),
		],
		default='GEO'
	)
	apply_modifiers: BoolProperty(
		name="Apply clip and section",
		description="Export the points left by the modifiers (clip and section) instead of all the stored points",
		default=True
	)
	export_attributes: BoolProperty(
		name="Attributes",
		description="Also export the point attributes, such as intensity or classification",
		default=True
	)

	@classmethod
	def poll(cls, context):
		return any(hasPoints(obj) for obj in context.selected_objects)

	def draw(self, context):
		layout = self.layout
		layout.prop(self, 'coordinates')
		if self.coordinates == 'GEO' and not GeoScene(context.scene).isGeoref:
			layout.label(text="Scene not georeferenced: scene coordinates", icon='ERROR')
		layout.prop(self, 'apply_modifiers')
		layout.prop(self, 'export_attributes')

	def execute(self, context):
		objs = [obj for obj in context.selected_objects if hasPoints(obj)]
		geoscn = GeoScene(context.scene)
		georef = self.coordinates == 'GEO' and geoscn.isGeoref
		if self.coordinates == 'GEO' and not georef:
			self.report({'WARNING'}, "Scene is not georeferenced, exporting scene coordinates")

		depsgraph = context.evaluated_depsgraph_get()
		clouds = [readPoints(obj, depsgraph, self.apply_modifiers, self.export_attributes) for obj in objs]
		xyz = np.concatenate([cloud[0] for cloud in clouds])
		if georef:
			dx, dy = geoscn.getOriginPrj()
			xyz[:, 0] += dx
			xyz[:, 1] += dy
		fields = {}
		dtype = np.float64 if georef else np.float32
		for i, axis in enumerate('xyz'):
			fields[axis] = xyz[:, i].astype(dtype)

		#merging several clouds: keep the colors and attributes that all of them have
		if all(cloud[1] is not None for cloud in clouds):
			colors = np.concatenate([cloud[1] for cloud in clouds])
			for i, channel in enumerate(('red', 'green', 'blue')):
				fields[channel] = colors[:, i]
		names = [name for name, values in clouds[0][2].items()
			if all(name in cloud[2] and cloud[2][name].dtype == values.dtype for cloud in clouds)]
		dropped = {name for cloud in clouds for name in cloud[2]} - set(names)
		for name in names:
			if name in fields:
				continue #an attribute named like x or red
			fields[name] = np.concatenate([cloud[2][name] for cloud in clouds])

		comments = ['Exported by BlenderGIS from ' + ', '.join(obj.name for obj in objs)]
		if georef:
			comments.append('crs ' + geoscn.crs)
		try:
			count = writePly(self.filepath, fields, comments)
		except (OSError, ValueError) as e:
			log.error("Cannot write PLY file", exc_info=True)
			self.report({'ERROR'}, f"Cannot write {self.filepath}: {e}")
			return {'CANCELLED'}
		if dropped:
			self.report({'WARNING'}, "Attributes missing in some point clouds not exported: " + ', '.join(sorted(dropped)))
		self.report({'INFO'}, f"Exported {count} points to {self.filepath}")
		return {'FINISHED'}


def register():
	try:
		bpy.utils.register_class(EXPORTGIS_OT_ply)
	except ValueError:
		log.warning('{} is already registered, now unregister and retry... '.format(EXPORTGIS_OT_ply))
		bpy.utils.unregister_class(EXPORTGIS_OT_ply)
		bpy.utils.register_class(EXPORTGIS_OT_ply)

def unregister():
	bpy.utils.unregister_class(EXPORTGIS_OT_ply)
