import math

import bpy
import numpy as np

import logging
log = logging.getLogger(__name__)

#shared geometry nodes group reused by every point cloud import, never edit it for a single object
#(renamed from 'Point Cloud Visualisation Node' when the material switched to the per point color attribute)
POINTCLOUD_NODE_GROUP = 'BGIS Point Cloud'
POINTCLOUD_MATERIAL = 'BGIS_PointCloud'
POINTCLOUD_RASTER_MATERIAL = 'BGIS_PointCloud_Raster'
COLOR_ATTRIBUTE = 'Col' #per point color (LAS RGB or baked raster)
UV_ATTRIBUTE = 'geo_uv_texture' #per point raster uv coordinates
FALLBACK_COLOR = (0.6, 0.6, 0.6, 1.0) #color of points without color data
#shared geometry nodes group clipping / sectioning any point cloud (mesh vertices or point cloud object),
#used by its own modifier at the top of the stack, the inputs are set per modifier
POINT_FILTER_NODE_GROUP = 'BGIS Point Filter'
CLIP_OBJECT_INPUT = 'Clip Object' #closed mesh cutting the points, per modifier
CLIP_KEEP_INSIDE_INPUT = 'Keep Inside' #keep the points inside the clip object instead of removing them
SECTION_OBJECT_INPUT = 'Section Object' #plane (local XY) of the section slab, per modifier
SECTION_THICKNESS_INPUT = 'Section Thickness'


def _socket(sockets, identifier):
	'''Get a socket by identifier, Mix node sockets share their names across data types'''
	return next(s for s in sockets if s.identifier == identifier)

def _greyFallbackShader(nodes, links, colorSocket, alphaSocket):
	'''Principled shader showing colorSocket, or a neutral grey where alphaSocket is 0 (no color data)'''
	mix = nodes.new('ShaderNodeMix')
	mix.data_type = 'RGBA'
	_socket(mix.inputs, 'A_Color').default_value = FALLBACK_COLOR
	shader = nodes.new('ShaderNodeBsdfPrincipled')
	output = nodes.new('ShaderNodeOutputMaterial')
	links.new(alphaSocket, _socket(mix.inputs, 'Factor_Float'))
	links.new(colorSocket, _socket(mix.inputs, 'B_Color'))
	links.new(_socket(mix.outputs, 'Result_Color'), shader.inputs['Base Color'])
	links.new(shader.outputs['BSDF'], output.inputs['Surface'])
	return [mix, shader, output]

def _layoutNodes(nodes):
	x = 0
	for node in nodes:
		node.location = (x, 0)
		x += node.width + 50

def _newMaterial(name):
	mat = bpy.data.materials.new(name=name)
	mat.use_nodes = True
	mat.node_tree.nodes.clear()
	return mat

def getPointCloudMaterial():
	'''Shared material showing the per point color attribute (grey when missing)'''
	mat = bpy.data.materials.get(POINTCLOUD_MATERIAL)
	if mat is not None:
		return mat
	mat = _newMaterial(POINTCLOUD_MATERIAL)
	nodes, links = mat.node_tree.nodes, mat.node_tree.links
	attr = nodes.new('ShaderNodeAttribute')
	attr.attribute_type = 'GEOMETRY'
	attr.attribute_name = COLOR_ATTRIBUTE
	#a missing attribute reads as 0, so its alpha drives the grey fallback
	_layoutNodes([attr] + _greyFallbackShader(nodes, links, attr.outputs['Color'], attr.outputs['Alpha']))
	return mat

def newRasterPointCloudMaterial(image):
	'''Material projecting image on points through the raster uv attribute (grey outside the raster)'''
	mat = _newMaterial(POINTCLOUD_RASTER_MATERIAL + '.' + image.name)
	nodes, links = mat.node_tree.nodes, mat.node_tree.links
	attr = nodes.new('ShaderNodeAttribute')
	attr.attribute_type = 'GEOMETRY'
	attr.attribute_name = UV_ATTRIBUTE
	tex = nodes.new('ShaderNodeTexImage')
	tex.image = image
	tex.extension = 'CLIP' #transparent outside the raster extent
	links.new(attr.outputs['Vector'], tex.inputs['Vector'])
	_layoutNodes([attr, tex] + _greyFallbackShader(nodes, links, tex.outputs['Color'], tex.outputs['Alpha']))
	return mat

def setPointCloudMaterial(obj, mat, singleUser=True):
	'''
	Assign mat to obj. Point clouds imported by BlenderGIS get their material from the
	Set Material node of their geometry nodes modifier. With singleUser, the shared template
	group (or any group used elsewhere) is copied first, so other and future point clouds keep their material
	'''
	for modifier in obj.modifiers:
		if modifier.type != 'NODES' or modifier.node_group is None:
			continue
		setMatNodes = [node for node in modifier.node_group.nodes if node.type == 'SET_MATERIAL']
		if not setMatNodes:
			continue
		if all(node.inputs['Material'].default_value == mat for node in setMatNodes):
			return #already set, no need to make the group single user
		if singleUser and (modifier.node_group.name == POINTCLOUD_NODE_GROUP or modifier.node_group.users > 1):
			modifier.node_group = modifier.node_group.copy()
			setMatNodes = [node for node in modifier.node_group.nodes if node.type == 'SET_MATERIAL']
		for node in setMatNodes:
			node.inputs['Material'].default_value = mat
		return
	#plain mesh
	if obj.data.materials:
		obj.data.materials[0] = mat
	else:
		obj.data.materials.append(mat)

def groupInputIdentifier(node_group, name):
	'''Identifier (Socket_n) of a group input, used to reach its value on the modifier, or None'''
	for item in node_group.interface.items_tree:
		if item.item_type == 'SOCKET' and item.in_out == 'INPUT' and item.name == name:
			return item.identifier
	return None

def getModifierInput(mod, name):
	'''Value of a geometry nodes group input on the modifier'''
	identifier = groupInputIdentifier(mod.node_group, name)
	if hasattr(mod, "properties"): # Blender 5.x: inputs are no longer IDProperties of the modifier
		return getattr(mod.properties.inputs, identifier).value
	return mod.get(identifier)

def setModifierInput(mod, name, value):
	identifier = groupInputIdentifier(mod.node_group, name)
	if hasattr(mod, "properties"): # Blender 5.x
		getattr(mod.properties.inputs, identifier).value = value
	else:
		mod[identifier] = value
	mod.id_data.update_tag() #setting an IDProperty does not update the modifier

def pointFilterModifier(obj):
	'''Geometry nodes modifier clipping / sectioning the points of obj, or None'''
	return next((mod for mod in obj.modifiers
		if mod.type == 'NODES' and mod.node_group is not None and mod.node_group.name.startswith(POINT_FILTER_NODE_GROUP)), None)

def addPointFilterModifier(obj):
	'''Add the clip and section modifier first in the stack, so it filters the raw points before any display modifier'''
	mod = pointFilterModifier(obj)
	if mod is not None:
		return mod
	mod = obj.modifiers.new(POINT_FILTER_NODE_GROUP, 'NODES')
	mod.node_group = getPointFilterNodeGroup()
	obj.modifiers.move(len(obj.modifiers) - 1, 0)
	return mod

def addSectionHelper(obj):
	'''
	Empty driving the section of obj, parented to it so it follows the point cloud. It is drawn as a
	square in its local XY plane (a cube empty flattened in Z, the section ignores the empty scale),
	sized to cover the whole cloud, and starts as a vertical section through the cloud center.
	'''
	mod = addPointFilterModifier(obj)
	corners = np.array([tuple(corner) for corner in obj.bound_box])
	low, high = corners.min(axis=0), corners.max(axis=0)
	diagonal = float(np.linalg.norm(high - low)) or 1.0
	empty = bpy.data.objects.new(obj.name + ' Section', None)
	empty.empty_display_type = 'CUBE'
	empty.empty_display_size = diagonal / 2
	empty.scale = (1, 1, 0.001) #not zero, the section reads the empty rotation from its matrix
	empty.parent = obj
	empty.location = (low + high) / 2
	empty.rotation_euler = (math.pi / 2, 0, 0)
	for collection in obj.users_collection:
		collection.objects.link(empty)
	setModifierInput(mod, SECTION_OBJECT_INPUT, empty)
	setModifierInput(mod, SECTION_THICKNESS_INPUT, diagonal / 100)
	return empty

def getPointFilterNodeGroup():
	'''Shared clip and section group, both stages delete nothing until their object is set'''
	node_group = bpy.data.node_groups.get(POINT_FILTER_NODE_GROUP)
	if node_group is not None:
		return node_group
	node_group = bpy.data.node_groups.new(POINT_FILTER_NODE_GROUP, 'GeometryNodeTree')
	node_group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
	node_group.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
	groupIn = node_group.nodes.new('NodeGroupInput')
	groupOut = node_group.nodes.new('NodeGroupOutput')
	node_group.links.new(groupIn.outputs['Geometry'], groupOut.inputs['Geometry'])
	_addClipNodes(node_group)
	_addSectionNodes(node_group)
	groupOut.location = (max(node.location.x + node.width for node in node_group.nodes) + 50, 0)
	return node_group

def _insertDeleteStage(node_group, selection):
	'''Delete the points of selection from the geometry feeding the group output'''
	nodes, links = node_group.nodes, node_group.links
	groupOut = next(node for node in nodes if node.type == 'GROUP_OUTPUT')
	delete = nodes.new('GeometryNodeDeleteGeometry')
	delete.domain = 'POINT'
	links.new(groupOut.inputs['Geometry'].links[0].from_socket, delete.inputs['Geometry'])
	links.new(selection, delete.inputs['Selection'])
	links.new(delete.outputs['Geometry'], groupOut.inputs['Geometry'])
	return delete

def _layoutColumns(node_group, columns, offsetY):
	'''Place a filter stage below the group input, one column of nodes after the other'''
	groupIn = next(node for node in node_group.nodes if node.type == 'GROUP_INPUT')
	x, y = groupIn.location.x, groupIn.location.y - offsetY
	for column in columns:
		for row, node in enumerate(column):
			node.location = (x, y - row * 200)
		x += max(node.width for node in column) + 50

def _addClipNodes(node_group):
	'''
	Clip by mesh stage: points inside the Clip Object (or outside with Keep Inside) are deleted.
	A point is inside when it lies behind the nearest surface of the clip object, so the clip mesh
	must be closed with outward normals.
	'''
	nodes, links = node_group.nodes, node_group.links
	groupIn = next(node for node in nodes if node.type == 'GROUP_INPUT')
	node_group.interface.new_socket(CLIP_OBJECT_INPUT, in_out='INPUT', socket_type='NodeSocketObject')
	keepInside = node_group.interface.new_socket(CLIP_KEEP_INSIDE_INPUT, in_out='INPUT', socket_type='NodeSocketBool')
	keepInside.default_value = False

	objInfo = nodes.new('GeometryNodeObjectInfo')
	objInfo.transform_space = 'RELATIVE' #clip mesh in the point cloud local space
	position = nodes.new('GeometryNodeInputPosition')
	normal = nodes.new('GeometryNodeInputNormal')
	samplePos = nodes.new('GeometryNodeSampleNearestSurface')
	sampleNormal = nodes.new('GeometryNodeSampleNearestSurface')
	toPoint = nodes.new('ShaderNodeVectorMath')
	toPoint.operation = 'SUBTRACT'
	dot = nodes.new('ShaderNodeVectorMath')
	dot.operation = 'DOT_PRODUCT'
	behind = nodes.new('FunctionNodeCompare')
	behind.data_type = 'FLOAT'
	behind.operation = 'LESS_THAN'
	behind.inputs['B'].default_value = 0
	invert = nodes.new('FunctionNodeBooleanMath')
	invert.operation = 'XOR'
	valid = nodes.new('FunctionNodeBooleanMath') #no clip object: empty mesh, sampling is not valid
	valid.operation = 'AND'

	links.new(groupIn.outputs[CLIP_OBJECT_INPUT], objInfo.inputs['Object'])
	for sample, field in ((samplePos, position.outputs['Position']), (sampleNormal, normal.outputs['Normal'])):
		sample.data_type = 'FLOAT_VECTOR'
		links.new(objInfo.outputs['Geometry'], sample.inputs['Mesh'])
		links.new(field, sample.inputs['Value'])
	links.new(position.outputs['Position'], toPoint.inputs[0])
	links.new(samplePos.outputs['Value'], toPoint.inputs[1])
	links.new(toPoint.outputs['Vector'], dot.inputs[0])
	links.new(sampleNormal.outputs['Value'], dot.inputs[1])
	links.new(dot.outputs['Value'], behind.inputs['A'])
	links.new(behind.outputs['Result'], invert.inputs[0])
	links.new(groupIn.outputs[CLIP_KEEP_INSIDE_INPUT], invert.inputs[1])
	links.new(samplePos.outputs['Is Valid'], valid.inputs[0])
	links.new(invert.outputs['Boolean'], valid.inputs[1])
	delete = _insertDeleteStage(node_group, valid.outputs['Boolean'])
	_layoutColumns(node_group, [[objInfo, position, normal], [samplePos, sampleNormal], [toPoint], [dot], [behind], [invert], [valid], [delete]], 0)

def _addSectionNodes(node_group):
	'''
	Section stage: only the points closer than Thickness / 2 to the plane of the Section Object
	(its local XY plane, the normal is its local Z axis) are kept. Scale is ignored, Thickness is in
	point cloud units.
	'''
	nodes, links = node_group.nodes, node_group.links
	groupIn = next(node for node in nodes if node.type == 'GROUP_INPUT')
	node_group.interface.new_socket(SECTION_OBJECT_INPUT, in_out='INPUT', socket_type='NodeSocketObject')
	thickness = node_group.interface.new_socket(SECTION_THICKNESS_INPUT, in_out='INPUT', socket_type='NodeSocketFloat')
	thickness.subtype = 'DISTANCE'
	thickness.default_value = 1.0
	thickness.min_value = 0.0

	objInfo = nodes.new('GeometryNodeObjectInfo')
	objInfo.transform_space = 'RELATIVE' #section plane in the point cloud local space
	position = nodes.new('GeometryNodeInputPosition')
	normal = nodes.new('FunctionNodeRotateVector')
	normal.inputs['Vector'].default_value = (0, 0, 1)
	toPoint = nodes.new('ShaderNodeVectorMath')
	toPoint.operation = 'SUBTRACT'
	dot = nodes.new('ShaderNodeVectorMath')
	dot.operation = 'DOT_PRODUCT'
	distance = nodes.new('ShaderNodeMath')
	distance.operation = 'ABSOLUTE'
	halfThickness = nodes.new('ShaderNodeMath')
	halfThickness.operation = 'MULTIPLY'
	halfThickness.inputs[1].default_value = 0.5
	outside = nodes.new('FunctionNodeCompare')
	outside.data_type = 'FLOAT'
	outside.operation = 'GREATER_THAN'
	scale = nodes.new('ShaderNodeVectorMath') #no section object: Object Info outputs a zero scale
	scale.operation = 'LENGTH'
	hasObject = nodes.new('FunctionNodeCompare')
	hasObject.data_type = 'FLOAT'
	hasObject.operation = 'GREATER_THAN'
	hasObject.inputs['B'].default_value = 0
	valid = nodes.new('FunctionNodeBooleanMath')
	valid.operation = 'AND'

	links.new(groupIn.outputs[SECTION_OBJECT_INPUT], objInfo.inputs['Object'])
	links.new(objInfo.outputs['Rotation'], normal.inputs['Rotation'])
	links.new(position.outputs['Position'], toPoint.inputs[0])
	links.new(objInfo.outputs['Location'], toPoint.inputs[1])
	links.new(toPoint.outputs['Vector'], dot.inputs[0])
	links.new(normal.outputs['Vector'], dot.inputs[1])
	links.new(dot.outputs['Value'], distance.inputs[0])
	links.new(groupIn.outputs[SECTION_THICKNESS_INPUT], halfThickness.inputs[0])
	links.new(distance.outputs['Value'], outside.inputs['A'])
	links.new(halfThickness.outputs['Value'], outside.inputs['B'])
	links.new(objInfo.outputs['Scale'], scale.inputs[0])
	links.new(scale.outputs['Value'], hasObject.inputs['A'])
	links.new(hasObject.outputs['Result'], valid.inputs[0])
	links.new(outside.outputs['Result'], valid.inputs[1])
	delete = _insertDeleteStage(node_group, valid.outputs['Boolean'])
	_layoutColumns(node_group, [[objInfo, position], [normal, toPoint], [dot, scale], [distance, halfThickness], [outside, hasObject], [valid], [delete]], 700)

def setPointColors(mesh, rgba):
	'''Write a (n, 4) float array of sRGB encoded colors to the per point color attribute'''
	attr = mesh.color_attributes.get(COLOR_ATTRIBUTE)
	if attr is not None and (attr.domain != 'POINT' or attr.data_type != 'BYTE_COLOR'):
		mesh.color_attributes.remove(attr)
		attr = None
	if attr is None:
		attr = mesh.color_attributes.new(COLOR_ATTRIBUTE, 'BYTE_COLOR', 'POINT')
	attr.data.foreach_set('color_srgb', np.ascontiguousarray(rgba, dtype=np.float32).ravel())

def getPointColors(mesh):
	'''Read the per point color attribute as a (n, 4) sRGB float array, or None'''
	attr = mesh.color_attributes.get(COLOR_ATTRIBUTE)
	if attr is None or attr.domain != 'POINT':
		return None
	rgba = np.empty(len(mesh.vertices) * 4, dtype=np.float32)
	attr.data.foreach_get('color_srgb', rgba)
	return rgba.reshape(-1, 4)
