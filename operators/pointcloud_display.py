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

'''Point cloud display settings: choose which data (RGB, elevation, LAS attribute, raster) colors the points'''

import logging
log = logging.getLogger(__name__)

import bpy
import numpy as np
from bpy.props import StringProperty, EnumProperty, FloatProperty, PointerProperty
from bpy.types import PropertyGroup, Panel

from .utils import getPointCloudMaterial, setPointCloudMaterial
from .utils.pointcloud_utils import POINTCLOUD_NODE_GROUP, COLOR_ATTRIBUTE, UV_ATTRIBUTE, CLIP_OBJECT_INPUT, CLIP_KEEP_INSIDE_INPUT
from .utils.pointcloud_utils import SECTION_OBJECT_INPUT, SECTION_THICKNESS_INPUT
from .utils.pointcloud_utils import addPointFilterModifier, pointFilterModifier, groupInputIdentifier, getModifierInput, addSectionHelper

ATTRIBUTE_MATERIAL = 'BGIS_PointCloud_Attr'
NUMERIC_TYPES = {'FLOAT', 'INT', 'INT8', 'BOOLEAN'}
#internal or color attributes, not meaningful as a scalar to color by
SKIPPED_ATTRIBUTES = {'position', COLOR_ATTRIBUTE, UV_ATTRIBUTE, 'red', 'green', 'blue'}

#ASPRS LAS 1.4 standard classes (sRGB)
CLASSIFICATION_COLORS = {
	0: (0.60, 0.60, 0.60), #created, never classified
	1: (0.80, 0.80, 0.80), #unclassified
	2: (0.65, 0.50, 0.35), #ground
	3: (0.65, 0.88, 0.45), #low vegetation
	4: (0.35, 0.72, 0.30), #medium vegetation
	5: (0.13, 0.50, 0.18), #high vegetation
	6: (0.88, 0.32, 0.26), #building
	7: (1.00, 0.00, 1.00), #low point (noise)
	8: (0.90, 0.90, 0.25), #reserved (model key-point)
	9: (0.22, 0.48, 0.92), #water
	10: (0.55, 0.32, 0.65), #rail
	11: (0.38, 0.38, 0.38), #road surface
	12: (0.92, 0.72, 0.22), #reserved (overlap)
	13: (1.00, 0.86, 0.25), #wire guard (shield)
	14: (1.00, 0.62, 0.12), #wire conductor (phase)
	15: (0.72, 0.22, 0.22), #transmission tower
	16: (0.95, 0.52, 0.32), #wire-structure connector
	17: (0.58, 0.52, 0.46), #bridge deck
	18: (1.00, 0.00, 0.60), #high noise
}
CLASSIFICATION_OTHER = (0.30, 0.30, 0.30) #user definable / reserved classes above 18
CLASSIFICATION_SLOTS = 32 #classes are mapped to [0, 1] in steps of 1/32

#color ramps (sRGB stops)
RAMPS = {
	'VIRIDIS': [(0.267, 0.005, 0.329), (0.231, 0.322, 0.545), (0.129, 0.569, 0.549), (0.369, 0.788, 0.384), (0.993, 0.906, 0.144)],
	'TURBO': [(0.190, 0.072, 0.232), (0.157, 0.565, 0.996), (0.380, 0.988, 0.424), (0.988, 0.729, 0.224), (0.478, 0.016, 0.012)],
	'GREY': [(0.05, 0.05, 0.05), (0.95, 0.95, 0.95)],
}


def srgbToLinear(c):
	c = np.asarray(c, dtype=np.float64)
	return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def pointCloudModifier(obj):
	'''Geometry nodes modifier displaying a point cloud imported by BlenderGIS, or None'''
	if obj is None or obj.type != 'MESH':
		return None
	return next((mod for mod in obj.modifiers
		if mod.type == 'NODES' and mod.node_group is not None and mod.node_group.name.startswith(POINTCLOUD_NODE_GROUP)), None)

def isPointCloud(obj):
	'''Point clouds imported by BlenderGIS are meshes displayed through the point cloud geometry nodes group'''
	return pointCloudModifier(obj) is not None

def numericAttributes(mesh):
	return [attr.name for attr in mesh.attributes
		if attr.domain == 'POINT' and attr.data_type in NUMERIC_TYPES
		and not attr.name.startswith('.') and attr.name not in SKIPPED_ATTRIBUTES]

def attributeValues(mesh, source, maxSamples=1_000_000):
	'''Values of the color source (an attribute name or ELEVATION) for every point, subsampled for large clouds'''
	n = len(mesh.vertices)
	step = max(1, n // maxSamples)
	if source == 'ELEVATION':
		co = np.empty(n * 3, dtype=np.float32)
		mesh.vertices.foreach_get('co', co)
		return co[2::3][::step]
	attr = mesh.attributes[source]
	dtype = np.bool_ if attr.data_type == 'BOOLEAN' else (np.float32 if attr.data_type == 'FLOAT' else np.int32)
	values = np.empty(n, dtype=dtype)
	attr.data.foreach_get('value', values)
	return values[::step].astype(np.float32)

def percentileRange(mesh, source, low=2, high=98):
	values = attributeValues(mesh, source)
	if values.size == 0:
		return 0.0, 1.0
	vmin, vmax = np.percentile(values, [low, high])
	if vmax <= vmin: #constant attribute
		vmax = vmin + 1
	return float(vmin), float(vmax)


def _setRampStops(ramp, stops, interpolation='LINEAR'):
	'''Replace the stops of a color ramp, stops is a list of (position, linear rgb)'''
	elements = ramp.elements
	while len(elements) > 1:
		elements.remove(elements[-1])
	elements[0].position = stops[0][0]
	elements[0].color = (*stops[0][1], 1.0)
	for position, rgb in stops[1:]:
		elements.new(position).color = (*rgb, 1.0)
	ramp.interpolation = interpolation

def getAttributeMaterial(obj):
	'''Per object material coloring the points from a scalar (attribute or elevation) through a color ramp'''
	name = ATTRIBUTE_MATERIAL + '.' + obj.name
	mat = bpy.data.materials.get(name)
	if mat is not None:
		return mat
	mat = bpy.data.materials.new(name)
	mat.use_nodes = True
	nodes, links = mat.node_tree.nodes, mat.node_tree.links
	nodes.clear()
	attr = nodes.new('ShaderNodeAttribute')
	attr.name = 'Attribute'
	attr.attribute_type = 'GEOMETRY'
	texco = nodes.new('ShaderNodeTexCoord')
	texco.name = 'Texture Coordinate'
	sepz = nodes.new('ShaderNodeSeparateXYZ')
	sepz.name = 'Elevation'
	maprange = nodes.new('ShaderNodeMapRange')
	maprange.name = 'Map Range'
	maprange.clamp = True
	ramp = nodes.new('ShaderNodeValToRGB')
	ramp.name = 'Color Ramp'
	shader = nodes.new('ShaderNodeBsdfPrincipled')
	output = nodes.new('ShaderNodeOutputMaterial')
	links.new(texco.outputs['Object'], sepz.inputs['Vector'])
	links.new(maprange.outputs['Result'], ramp.inputs['Fac'])
	links.new(ramp.outputs['Color'], shader.inputs['Base Color'])
	links.new(shader.outputs['BSDF'], output.inputs['Surface'])
	for i, node in enumerate([texco, sepz, attr, maprange, ramp, shader, output]):
		node.location = (i * 220, 0 if node not in (texco, sepz) else -250)
	return mat

def applyColorBy(obj):
	'''Update the material of a point cloud from its display settings'''
	settings = obj.bgis_pointcloud
	source = settings.colorByValue
	if source == 'RGB':
		setPointCloudMaterial(obj, getPointCloudMaterial())
		return
	if source == 'RASTER':
		if settings.rasterMaterial is not None:
			setPointCloudMaterial(obj, settings.rasterMaterial)
		return

	mat = getAttributeMaterial(obj)
	nodes, links = mat.node_tree.nodes, mat.node_tree.links
	maprange, ramp = nodes['Map Range'], nodes['Color Ramp'].color_ramp
	for link in list(maprange.inputs['Value'].links):
		links.remove(link)
	if source == 'ELEVATION':
		links.new(nodes['Elevation'].outputs['Z'], maprange.inputs['Value'])
	else:
		nodes['Attribute'].attribute_name = source
		links.new(nodes['Attribute'].outputs['Fac'], maprange.inputs['Value'])

	if source == 'classification':
		#one flat color per class value: class c is mapped to (c + 0.5) / 32, its stop sits at c / 32
		maprange.inputs['From Min'].default_value = -0.5
		maprange.inputs['From Max'].default_value = CLASSIFICATION_SLOTS - 0.5
		stops = [(c / CLASSIFICATION_SLOTS, srgbToLinear(rgb)) for c, rgb in sorted(CLASSIFICATION_COLORS.items())]
		stops.append(((max(CLASSIFICATION_COLORS) + 1) / CLASSIFICATION_SLOTS, srgbToLinear(CLASSIFICATION_OTHER)))
		_setRampStops(ramp, stops, 'CONSTANT')
	else:
		maprange.inputs['From Min'].default_value = settings.rangeMin
		maprange.inputs['From Max'].default_value = settings.rangeMax
		colors = RAMPS[settings.ramp]
		_setRampStops(ramp, [(i / (len(colors) - 1), srgbToLinear(rgb)) for i, rgb in enumerate(colors)])
	setPointCloudMaterial(obj, mat)


def setColorBy(obj, source):
	'''Select the color source of a point cloud (RGB, ELEVATION, RASTER or an attribute name) and apply it'''
	settings = obj.bgis_pointcloud
	settings.colorByValue = source
	if source not in ('RGB', 'RASTER', 'classification'):
		global _suspendRangeUpdate
		_suspendRangeUpdate = True #apply once below, not on each range assignment
		try:
			settings.rangeMin, settings.rangeMax = percentileRange(obj.data, source)
		finally:
			_suspendRangeUpdate = False
	applyColorBy(obj)


#Blender needs the strings of dynamic enum items to stay referenced
_colorByItems = []

def listColorBy(self, context):
	obj = self.id_data
	items = []
	if COLOR_ATTRIBUTE in obj.data.color_attributes:
		items.append(('RGB', 'RGB', "Point colors (LAS RGB or baked raster)"))
	if self.rasterMaterial is not None and UV_ATTRIBUTE in obj.data.attributes:
		items.append(('RASTER', 'Raster', "Raster projected with Colorize points from raster (Live)"))
	items.append(('ELEVATION', 'Elevation', "Point height"))
	for name in numericAttributes(obj.data):
		label = name.replace('_', ' ').capitalize()
		items.append((name, label, f"Color by the {name} attribute"))
	_colorByItems[:] = [(identifier, label, desc, i) for i, (identifier, label, desc) in enumerate(items)]
	return _colorByItems

def getColorBy(self):
	for identifier, _, _, value in listColorBy(self, None):
		if identifier == self.colorByValue:
			return value
	return 0

def setColorByEnum(self, value):
	for identifier, _, _, v in listColorBy(self, None):
		if v == value:
			setColorBy(self.id_data, identifier)
			return

_suspendRangeUpdate = False

def updateRange(self, context):
	if not _suspendRangeUpdate:
		applyColorBy(self.id_data)


class BGIS_PG_pointcloud(PropertyGroup):
	#the selected source is stored as a string, the enum index of dynamic items is not stable
	colorByValue: StringProperty(default='RGB')
	colorBy: EnumProperty(name="Color by", description="Data used to color the points", items=listColorBy, get=getColorBy, set=setColorByEnum)
	rangeMin: FloatProperty(name="Min", description="Value mapped to the start of the color ramp", update=updateRange)
	rangeMax: FloatProperty(name="Max", description="Value mapped to the end of the color ramp", default=1, update=updateRange)
	ramp: EnumProperty(
		name="Ramp",
		items=[('VIRIDIS', 'Viridis', ""), ('TURBO', 'Turbo', ""), ('GREY', 'Greyscale', "")],
		update=updateRange
		)
	rasterMaterial: PointerProperty(type=bpy.types.Material, description="Material of the raster projected with Colorize points from raster")


class BGIS_OT_pointcloud_reset_range(bpy.types.Operator):
	bl_idname = "bgis.pointcloud_reset_range"
	bl_label = "Reset range"
	bl_description = "Set the range to the 2nd - 98th percentile of the values"
	bl_options = {'UNDO', 'INTERNAL'}

	def execute(self, context):
		obj = context.active_object
		setColorBy(obj, obj.bgis_pointcloud.colorByValue)
		return {'FINISHED'}


def hasPoints(obj):
	'''Any point cloud: a point cloud object or a mesh made of vertices only (LAS, PLY, gaussian splats...)'''
	if obj is None:
		return False
	if obj.type == 'POINTCLOUD':
		return True
	return obj.type == 'MESH' and (len(obj.data.polygons) == 0 or pointFilterModifier(obj) is not None)


def pointFilterTarget(obj):
	'''Object whose clip and section settings are shown: a point cloud, or the point cloud a section helper drives'''
	if hasPoints(obj):
		return obj
	if obj is not None and obj.type == 'EMPTY' and hasPoints(obj.parent):
		mod = pointFilterModifier(obj.parent)
		if mod is not None and getModifierInput(mod, SECTION_OBJECT_INPUT) == obj:
			return obj.parent
	return None


class BGIS_OT_add_point_filter(bpy.types.Operator):
	bl_idname = "bgis.add_point_filter"
	bl_label = "Add clip and section"
	bl_description = "Add a modifier, first in the stack, to clip the points by a mesh or show a section slab"
	bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

	@classmethod
	def poll(cls, context):
		return hasPoints(context.active_object)

	def execute(self, context):
		addPointFilterModifier(context.active_object)
		return {'FINISHED'}


class BGIS_OT_add_section_helper(bpy.types.Operator):
	bl_idname = "bgis.add_section_helper"
	bl_label = "Add section helper"
	bl_description = "Add an empty, parented to the point cloud, whose plane drives the section. Rotate or move it to sweep the section"
	bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

	@classmethod
	def poll(cls, context):
		return pointFilterTarget(context.active_object) is not None

	def execute(self, context):
		empty = addSectionHelper(pointFilterTarget(context.active_object))
		#select the helper so it can be rotated right away, the panel keeps showing its point cloud settings
		for obj in context.selected_objects:
			obj.select_set(False)
		empty.select_set(True)
		context.view_layer.objects.active = empty
		return {'FINISHED'}


def drawModifierInput(layout, mod, name, text):
	'''Draw the value of a geometry nodes group input on the modifier'''
	identifier = groupInputIdentifier(mod.node_group, name)
	if identifier is None:
		return
	if hasattr(mod, "properties"): # Blender 5.x
		socket = getattr(mod.properties.inputs, identifier, None)
		if socket is not None:
			layout.prop(socket, 'value', text=text)
	elif identifier in mod:
		layout.prop(mod, '["{}"]'.format(identifier), text=text)


class BGIS_PT_pointcloud(Panel):
	bl_category = "GIS"
	bl_label = "Point Cloud"
	bl_space_type = "VIEW_3D"
	bl_region_type = "UI"

	@classmethod
	def poll(cls, context):
		return isPointCloud(context.active_object)

	def draw(self, context):
		layout = self.layout
		obj = context.active_object
		settings = obj.bgis_pointcloud
		layout.prop(settings, 'colorBy')
		source = settings.colorByValue
		if source not in ('RGB', 'RASTER', 'classification'):
			row = layout.row(align=True)
			row.prop(settings, 'rangeMin')
			row.prop(settings, 'rangeMax')
			row.operator('bgis.pointcloud_reset_range', text='', icon='FILE_REFRESH')
			layout.prop(settings, 'ramp')
		#inputs of the point cloud geometry nodes group
		mod = pointCloudModifier(obj)
		drawModifierInput(layout, mod, 'Radius', "Point size")
		layout.operator('object.colorize_points', icon='IMAGE_DATA')


class BGIS_PT_point_filter(Panel):
	bl_category = "GIS"
	bl_label = "Clip and Section"
	bl_space_type = "VIEW_3D"
	bl_region_type = "UI"

	@classmethod
	def poll(cls, context):
		return pointFilterTarget(context.active_object) is not None

	def draw(self, context):
		layout = self.layout
		target = pointFilterTarget(context.active_object)
		if target != context.active_object:
			layout.label(text=target.name, icon='OUTLINER_OB_POINTCLOUD')
		mod = pointFilterModifier(target)
		if mod is None:
			layout.operator('bgis.add_point_filter', icon='MOD_BOOLEAN')
			layout.operator('bgis.add_section_helper', icon='EMPTY_DATA')
			return
		drawModifierInput(layout, mod, CLIP_OBJECT_INPUT, "Clip by")
		drawModifierInput(layout, mod, CLIP_KEEP_INSIDE_INPUT, "Keep inside")
		drawModifierInput(layout, mod, SECTION_OBJECT_INPUT, "Section by")
		drawModifierInput(layout, mod, SECTION_THICKNESS_INPUT, "Thickness")
		layout.operator('bgis.add_section_helper', icon='EMPTY_DATA')


classes = [BGIS_PG_pointcloud, BGIS_OT_pointcloud_reset_range, BGIS_OT_add_point_filter, BGIS_OT_add_section_helper, BGIS_PT_pointcloud, BGIS_PT_point_filter]

def register():
	for cls in classes:
		try:
			bpy.utils.register_class(cls)
		except ValueError:
			log.warning('{} is already registered, now unregister and retry... '.format(cls))
			bpy.utils.unregister_class(cls)
			bpy.utils.register_class(cls)
	bpy.types.Object.bgis_pointcloud = PointerProperty(type=BGIS_PG_pointcloud)

def unregister():
	del bpy.types.Object.bgis_pointcloud
	for cls in reversed(classes):
		bpy.utils.unregister_class(cls)
