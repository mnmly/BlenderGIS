import bpy
import numpy as np

import logging
log = logging.getLogger(__name__)

POINTCLOUD_MATERIAL = 'BGIS_PointCloud'
POINTCLOUD_RASTER_MATERIAL = 'BGIS_PointCloud_Raster'
COLOR_ATTRIBUTE = 'Col' #per point color (LAS RGB or baked raster)
UV_ATTRIBUTE = 'geo_uv_texture' #per point raster uv coordinates
FALLBACK_COLOR = (0.6, 0.6, 0.6, 1.0) #color of points without color data


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

def setPointCloudMaterial(obj, mat):
	'''
	Assign mat to obj. Point clouds imported by BlenderGIS get their material from the
	Set Material node of their geometry nodes modifier, which is made single user
	so other point clouds keep their own material
	'''
	for modifier in obj.modifiers:
		if modifier.type != 'NODES' or modifier.node_group is None:
			continue
		setMatNodes = [node for node in modifier.node_group.nodes if node.type == 'SET_MATERIAL']
		if not setMatNodes:
			continue
		if modifier.node_group.users > 1:
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
