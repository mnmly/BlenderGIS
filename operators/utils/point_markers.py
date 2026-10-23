'''Show the vertices of point layers as spheres, with a shared geometry nodes group (instances, the points stay editable)'''

import math

import bpy

POINT_MARKERS_NODE_GROUP = 'BGIS Point Markers'
POINT_MARKERS_MODIFIER = 'Point Markers'


def getPointMarkersNodeGroup():
	node_group = bpy.data.node_groups.get(POINT_MARKERS_NODE_GROUP)
	if node_group is not None:
		return node_group
	node_group = bpy.data.node_groups.new(POINT_MARKERS_NODE_GROUP, 'GeometryNodeTree')
	interface = node_group.interface
	interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
	radius = interface.new_socket('Radius', in_out='INPUT', socket_type='NodeSocketFloat')
	radius.default_value, radius.min_value, radius.subtype = 1.0, 0.0, 'DISTANCE'
	detail = interface.new_socket('Detail', in_out='INPUT', socket_type='NodeSocketInt')
	detail.default_value, detail.min_value, detail.max_value = 2, 1, 5
	#instances do not use the material slots of the object, so the material is given to the spheres
	interface.new_socket('Material', in_out='INPUT', socket_type='NodeSocketMaterial')
	interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

	nodes, links = node_group.nodes, node_group.links
	group_in = nodes.new('NodeGroupInput')
	sphere = nodes.new('GeometryNodeMeshIcoSphere')
	smooth = nodes.new('GeometryNodeSetShadeSmooth')
	material = nodes.new('GeometryNodeSetMaterial')
	instance = nodes.new('GeometryNodeInstanceOnPoints')
	group_out = nodes.new('NodeGroupOutput')

	links.new(group_in.outputs['Radius'], sphere.inputs['Radius'])
	links.new(group_in.outputs['Detail'], sphere.inputs['Subdivisions'])
	links.new(sphere.outputs['Mesh'], smooth.inputs['Geometry'])
	links.new(smooth.outputs['Geometry'], material.inputs['Geometry'])
	links.new(group_in.outputs['Material'], material.inputs['Material'])
	links.new(group_in.outputs['Geometry'], instance.inputs['Points'])
	links.new(material.outputs['Geometry'], instance.inputs['Instance'])
	links.new(instance.outputs['Instances'], group_out.inputs['Geometry'])

	for x, node in enumerate((group_in, sphere, smooth, material, instance, group_out)):
		node.location = (x * 200, 0)
	sphere.location.y = group_in.location.y = -150
	return node_group

def setModifierInput(modifier, name, value):
	identifier = modifier.node_group.interface.items_tree[name].identifier
	if hasattr(modifier, 'properties'): #Blender 5.x: inputs are no longer IDProperties of the modifier
		getattr(modifier.properties.inputs, identifier).value = value
	else:
		modifier[identifier] = value

def markerRadius(objs):
	'''A radius that keeps the markers visible at the scale of the layer: 1/100 of its extent, 1 m at least'''
	xs, ys = [], []
	for obj in objs:
		for v in obj.data.vertices:
			co = obj.matrix_world @ v.co
			xs.append(co.x)
			ys.append(co.y)
	if not xs:
		return 1.0
	return max(1.0, math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / 100)

def addPointMarkers(objs, radius=None):
	'''Add the point markers modifier to mesh objects made of points, with a common radius'''
	objs = [obj for obj in objs if obj.type == 'MESH' and len(obj.data.vertices) and not len(obj.data.edges)]
	if not objs:
		return []
	if radius is None:
		radius = markerRadius(objs)
	node_group = getPointMarkersNodeGroup()
	for obj in objs:
		modifier = obj.modifiers.get(POINT_MARKERS_MODIFIER) or obj.modifiers.new(POINT_MARKERS_MODIFIER, 'NODES')
		modifier.node_group = node_group
		setModifierInput(modifier, 'Radius', radius)
		if obj.active_material is not None:
			setModifierInput(modifier, 'Material', obj.active_material)
	return objs

def setPointMarkersMaterial(obj, mat):
	modifier = obj.modifiers.get(POINT_MARKERS_MODIFIER)
	if modifier is not None and modifier.node_group is not None:
		setModifierInput(modifier, 'Material', mat)
