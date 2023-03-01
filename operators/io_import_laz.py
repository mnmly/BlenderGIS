
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

import bpy
import bmesh
import os
import sys
import math
from mathutils import Vector
import subprocess
from pathlib import Path
import numpy as np#Ship with Blender since 2.70

import logging
log = logging.getLogger(__name__)

from ..geoscene import GeoScene, georefManagerLayout
from ..prefs import PredefCRS

from ..core.georaster import GeoRaster
from .utils import bpyGeoRaster, exportAsMesh
from .utils import placeObj, adjust3Dview, showTextures, addTexture, getBBOX
from .utils import rasterExtentToMesh, geoRastUVmap, setDisplacer

from ..core import HAS_GDAL
if HAS_GDAL:
	from osgeo import gdal

from ..core import XY as xy
from ..core.errors import OverlapError
from ..core.proj import Reproj


from bpy_extras.io_utils import ImportHelper #helper class defines filename and invoke() function which calls the file selector
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty, CollectionProperty, FloatProperty
from bpy.types import Operator

py_path = Path(sys.prefix) / "bin"
py_exec = next(py_path.glob("python*"))

try:
	import laspy
except ImportError:
    subprocess.call([py_exec, "-m", "pip", "install", "laspy", "laszip"])

try:
	import pyproj
except ImportError:
    subprocess.call([py_exec, "-m", "pip", "install", "pyproj"])


PKG, SUBPKG = __package__.split('.', maxsplit=1)

class IMPORTLAZ_OT_georaster(Operator, ImportHelper):
	"""Import georeferenced raster (need world file)"""
	bl_idname = "importgis.laz"  # important since its how bpy.ops.importgis.laz is constructed (allows calling operator from python console or another script)
	#bl_idname rules: must contain one '.' (dot) charactere, no capital letters, no reserved words (like 'import')
	bl_description = 'Import LAZ/LAS with world file'
	bl_label = "Import LAZ/LAS"
	bl_options = {"UNDO"}

	def listObjects(self, context):
		#Function used to update the objects list (obj_list) used by the dropdown box.
		objs = [] #list containing tuples of each object
		for index, object in enumerate(bpy.context.scene.objects): #iterate over all objects
			if object.type == 'MESH':
				objs.append((str(index), object.name, "Object named " +object.name)) #put each object in a tuple (key, label, tooltip) and add this to the objects list
		return objs

	# ImportHelper class properties
	filter_glob: StringProperty(
			default="*.laz;*.laz",
			options={'HIDDEN'},
			)

	# Raster CRS definition
	def listPredefCRS(self, context):
		return PredefCRS.getEnumItems()

	files: CollectionProperty(
		type = bpy.types.OperatorFileListElement,
		options = {'HIDDEN', 'SKIP_SAVE'}
	)
	rastCRS: EnumProperty(
		name = "Raster CRS",
		description = "Choose a Coordinate Reference System",
		items = listPredefCRS,
		)
	fallbackCRS: EnumProperty(
		name = "Fallback CRS",
		description = "Choose a Coordinate Reference System when LIDAR data doesn't contain CRS metadata",
		items = listPredefCRS,
		)
	
	import_scale: FloatProperty(
		name = "Import Scale",
		description = "To use a custom import scale, change this value",
		default=1.0
	)

	auto_centre: BoolProperty(
		name = "Auto Centre On Import",
		description = "If enabled, automatically centers the cloud on import.",
		default=False
	)

	reprojection: BoolProperty(
			name="Specifiy raster CRS",
			description="Specifiy raster CRS if it's different from scene CRS",
			default=False )
	#
	objectsLst: EnumProperty(attr="obj_list", name="Objects", description="Choose object to edit", items=listObjects)
	#
	#Subdivise (as DEM option)
	def listSubdivisionModes(self, context):
		items = [ ('subsurf', 'Subsurf', "Add a subsurf modifier"), ('none', 'None', "No subdivision")]
		if not self.demOnMesh:
			#mesh subdivision method can not be applyed on an existing mesh
			#this option makes sense only when the mesh is created from scratch
			items.append(('mesh', 'Mesh', "Create vertices at each pixels"))
		return items

	subdivision: EnumProperty(
			name="Subdivision",
			description="How to subdivise the plane (dispacer needs vertex to work with)",
			items=listSubdivisionModes
			)
	#
	demOnMesh: BoolProperty(
			name="Apply on existing mesh",
			description="Use DEM as displacer for an existing mesh",
			default=False
			)
	#
	clip: BoolProperty(
			name="Clip to working extent",
			description="Use the reference bounding box to clip the DEM",
			default=False
			)
	#
	demInterpolation: BoolProperty(
			name="Smooth relief",
			description="Use texture interpolation to smooth the resulting terrain",
			default=True
			)
	#
	fillNodata: BoolProperty(
			name="Fill nodata values",
			description="Interpolate existing nodata values to get an usuable displacement texture",
			default=False
			)
	#
	step: IntProperty(name = "Step", default=1, description="Pixel step", min=1)

	buildFaces: BoolProperty(name="Build faces", default=True, description='Build quad faces connecting pixel point cloud')

	def draw(self, context):
		#Function used by blender to draw the panel.
		layout = self.layout
		scn = bpy.context.scene
		geoscn = GeoScene(scn)
		#
		if geoscn.isPartiallyGeoref:
			layout.prop(self, 'reprojection')
			if self.reprojection:
				self.crsInputLayout(context)
			#
			georefManagerLayout(self, context)
		else:
			self.crsInputLayout(context)
		self.fallbackCRSInputLayout(context)
		self.importScaleInputLayout(context)
		self.autoCentreInputLayout(context)

	def crsInputLayout(self, context):
		layout = self.layout
		row = layout.row(align=True)
		split = row.split(factor=0.35, align=True)
		split.label(text='CRS:')
		split.prop(self, "rastCRS", text='')
		row.operator("bgis.add_predef_crs", text='', icon='ADD')

	def fallbackCRSInputLayout(self, context):
		layout = self.layout
		row = layout.row(align=True)
		split = row.split(factor=0.35, align=True)
		split.label(text='Fallback CRS:')
		split.prop(self, "fallbackCRS", text='')
		row.operator("bgis.add_predef_crs", text='', icon='ADD')

	def importScaleInputLayout(self, context):
		layout = self.layout
		row = layout.row(align=True)
		row.prop(self, "import_scale")

	def autoCentreInputLayout(self, context):
		layout = self.layout
		row = layout.row(align=True)
		row.prop(self, "auto_centre")

	@classmethod
	def poll(cls, context):
		return context.mode == 'OBJECT'

	def execute(self, context):
		prefs = context.preferences.addons[PKG].preferences

		bpy.ops.object.select_all(action='DESELECT')
		#Get scene and some georef data
		scn = bpy.context.scene
		geoscn = GeoScene(scn)
		if geoscn.isBroken:
			self.report({'ERROR'}, "Scene georef is broken, please fix it beforehand")
			return {'CANCELLED'}

		scale = geoscn.scale #TODO

		if geoscn.isGeoref:
			dx, dy = geoscn.getOriginPrj()
			if self.reprojection:
				rastCRS = self.rastCRS
			else:
				rastCRS = geoscn.crs
		else: #if not geoscn.hasCRS
			rastCRS = self.rastCRS
			try:
				geoscn.crs = rastCRS
			except Exception as e:
				log.error("Cannot set scene crs", exc_info=True)
				self.report({'ERROR'}, "Cannot set scene crs, check logs for more infos")
				return {'CANCELLED'}

		#Raster reprojection throught UV mapping
		#build reprojector objects
		if geoscn.crs != rastCRS:
			rprj = True
			rprjToRaster = Reproj(geoscn.crs, rastCRS)
			rprjToScene = Reproj(rastCRS, geoscn.crs)
		else:
			rprj = False
			rprjToRaster = None
			rprjToScene = None

		common_prefix = os.path.commonprefix([f.name for f in self.files])
		parent_obj = bpy.data.objects.new(common_prefix, None)
		parent_obj.empty_display_type = 'PLAIN_AXES'   
		bpy.context.scene.collection.objects.link(parent_obj)
		bpy.context.view_layer.objects.active = parent_obj
		parent_obj.select_set(True)
		midpoint_list = []
		objects = []
		#Path
		for f in self.files:
			filePath = os.path.join(os.path.dirname(self.filepath), f.name)
			name = os.path.basename(filePath)[:-4]
			#Import
			try:
				las = laspy.read(filePath)
			except IOError as e:
				log.error("Unable to open raster", exc_info=True)
				self.report({'ERROR'}, "Unable to open raster, check logs for more infos")
				return {'CANCELLED'}
			except OverlapError:
				self.report({'ERROR'}, "Non overlap data")
				return {'CANCELLED'}

			pc = bpy.data.meshes.new("Point Cloud")
			verts, coords, source_crs, is_fallback = self.scaled_dimension(las, geoscn.crs if geoscn.hasCRS else None, self.fallbackCRS)
			verts = verts * self.import_scale
			min_coords = coords[0] * self.import_scale
			max_coords = coords[1] * self.import_scale
			midpoint = min_coords + (max_coords - min_coords) * 0.5
			midpoint_list.append(midpoint)
			delta_coords = max_coords - min_coords
			normalised_coords = (verts - min_coords - delta_coords / 2.0) / (delta_coords / 2.0)
			coords = normalised_coords * delta_coords / 2.0 # TODO FIX
			pc.from_pydata(coords, [], [])
			attribute_keys = ['classification', 
			#'return_number', 'number_of_returns', 'point_source_id', 'gps_time'
			]
			attribute_type = ['INT',
			#			'INT8', 		  'INT8', 			   'INT',			 'FLOAT'
			]
			for (i, key) in enumerate(attribute_keys):
				if key in las.point_format.dimension_names:
					pc.attributes.new(name=key, type=attribute_type[i], domain="POINT")
					pc.attributes[key].data.foreach_set("value", np.array(las[key]).tolist())
			obj = placeObj(pc, name)
			obj.location.x = midpoint[0]
			obj.location.y = midpoint[1]
			obj.location.z = midpoint[2]
			obj['laz_midpoint'] = midpoint
			obj['source_crs'] = source_crs.name
			obj['is_fallback'] = is_fallback
			obj['import_scale'] = self.import_scale
			self.assign_geometry_node(obj, context)
			objects.append(obj)

		midpoint_mean = np.mean(midpoint_list, axis=0)
		parent_obj.location.x = midpoint_mean[0]
		parent_obj.location.y = midpoint_mean[1]
		parent_obj.location.z = midpoint_mean[2]
		parent_obj['laz_midpoint'] = midpoint_mean

		bpy.context.evaluated_depsgraph_get().update()

		for obj in objects:
			# mat_world = obj.matrix_world.copy()
			obj.parent_type = 'OBJECT'
			obj.parent = parent_obj
			obj.matrix_parent_inverse = parent_obj.matrix_world.inverted()

		# bpy.context.evaluated_depsgraph_get().update()

		if not geoscn.isGeoref:
			dx, dy = midpoint_mean[0], midpoint_mean[1]
			geoscn.setOriginPrj(dx, dy)

		if geoscn.crsx != None:
			parent_obj.location.x -= geoscn.crsx
			parent_obj.location.y -= geoscn.crsy

		return {'FINISHED'}

	def assign_geometry_node(self, obj, context):
		node_group_name = "Point Cloud Visualisation Node"
		if node_group_name in bpy.data.node_groups:
			node_group = bpy.data.node_groups[node_group_name]
		else:
			material_pointcloud = self.new_pointcloud_material()
			node_group = self.new_pointcloud_geometry_node_group(context, material_pointcloud)
		modifier = obj.modifiers.new(node_group_name, 'NODES')
		modifier.node_group = node_group

	def place_node_alongside(self, node, other, padding, y = None):
		node.location.x = other.location.x + other.width + padding
		node.location.y = 0.0 if y == None else other.location.y
	
	def place_node_below(self, node, other, padding):
		node.location.x = other.location.x
		node.location.y = other.location.y - other.height - padding

	def new_pointcloud_material(self):
		material_id = 'M_PointCloud'
		mat = bpy.data.materials.get(material_id)
		if mat != None:
			return mat

		mat = bpy.data.materials.new(name=material_id)
		mat.use_nodes = True
		if mat.node_tree:
			mat.node_tree.links.clear()
			mat.node_tree.nodes.clear()
		
		nodes = mat.node_tree.nodes
		links = mat.node_tree.links

		node_attributes = nodes.new('ShaderNodeAttribute')
		node_texture = nodes.new('ShaderNodeTexImage')
		shader = nodes.new(type='ShaderNodeBsdfPrincipled')
		output = nodes.new(type='ShaderNodeOutputMaterial')

		node_attributes.attribute_type = 'GEOMETRY'
		node_attributes.attribute_name = 'geo_uv_texture'

		links.new(node_attributes.outputs['Vector'], node_texture.inputs['Vector'])
		links.new(node_texture.outputs['Color'], shader.inputs['Base Color'])
		links.new(shader.outputs['BSDF'], output.inputs['Surface'])

		# Set locations 
		padding = 50
		node_attributes.location.x = 0
		self.place_node_alongside(node_texture, node_attributes, padding)
		self.place_node_alongside(shader, node_texture, padding)
		self.place_node_alongside(output, shader, padding)

		for n in nodes:
			n.location.y = 0

	def node_for_type(self, sources, source_name, type_name):
		return [source for source in sources if source.type == type_name and source.name == source_name][0]

	def new_pointcloud_geometry_node_group(self, context, material_pointcloud):
		''' Create a new empty node group that can be used
			in a GeometryNodes modifier.
		'''
		node_group = bpy.data.node_groups.new('Point Cloud Visualisation Node', 'GeometryNodeTree')
		nodes = node_group.nodes
		links = node_group.links

		group_in = nodes.new('NodeGroupInput')
		group_out = nodes.new('NodeGroupOutput')
		node_mesh_to_points = nodes.new('GeometryNodeMeshToPoints')
		node_set_material = nodes.new('GeometryNodeSetMaterial')
		node_store_named_attribute = nodes.new('GeometryNodeStoreNamedAttribute')

		node_self_object = nodes.new('GeometryNodeSelfObject')
		node_self_position = nodes.new('GeometryNodeInputPosition')
		node_self_object_info = nodes.new('GeometryNodeObjectInfo')
		node_self_add = nodes.new('ShaderNodeVectorMath')
		node_self_add.operation = 'ADD'

		node_ref_object_info = nodes.new('GeometryNodeObjectInfo')
		node_ref_position = nodes.new('GeometryNodeInputPosition')
		node_ref_stats = nodes.new('GeometryNodeAttributeStatistic')
		node_ref_stats.data_type = 'FLOAT_VECTOR'
		node_ref_stats.domain = 'POINT'

		node_subtract_01 = nodes.new('ShaderNodeVectorMath')
		node_subtract_02 = nodes.new('ShaderNodeVectorMath')
		node_divide = nodes.new('ShaderNodeVectorMath')
		node_subtract_01.operation = 'SUBTRACT'
		node_subtract_02.operation = 'SUBTRACT'
		node_divide.operation = 'DIVIDE'

		# Set properties
		node_group.outputs.new('NodeSocketGeometry', 'Geometry')
		node_group.inputs.new('NodeSocketGeometry', 'Geometry')
		node_mesh_to_points.inputs['Radius'].default_value = 0.5

		node_store_named_attribute.data_type = 'FLOAT_VECTOR'
		node_store_named_attribute.domain = 'POINT'

		# Set links
		### Geometry
		links.new(group_in.outputs['Geometry'], node_mesh_to_points.inputs['Mesh'])
		links.new(node_mesh_to_points.outputs['Points'], node_set_material.inputs['Geometry'])
		links.new(node_set_material.outputs['Geometry'], node_store_named_attribute.inputs['Geometry'])
		node_set_material.inputs['Material'].default_value = material_pointcloud
		links.new(node_store_named_attribute.outputs['Geometry'], group_out.inputs['Geometry'])

		### Store Value
		#### Get self position including its current location of object itself
		links.new(node_self_object.outputs['Self Object'], node_self_object_info.inputs['Object'])
		links.new(node_self_object_info.outputs['Location'], node_self_add.inputs[0])
		links.new(node_self_position.outputs['Position'], node_self_add.inputs[1])

		#### Get reference object's stats and calculate uv values
		links.new(node_ref_object_info.outputs['Geometry'], node_ref_stats.inputs['Geometry'])
		links.new(node_ref_position.outputs['Position'], self.node_for_type(node_ref_stats.inputs, 'Attribute', 'VECTOR'))

		#### Bring self position to 0 relative to reference
		links.new(node_self_add.outputs['Vector'], node_subtract_01.inputs[0])
		links.new(self.node_for_type(node_ref_stats.outputs, 'Min', 'VECTOR'), node_subtract_01.inputs[1])

		#### Get delta of min max of reference
		links.new(self.node_for_type(node_ref_stats.outputs, 'Max', 'VECTOR'), node_subtract_02.inputs[0])
		links.new(self.node_for_type(node_ref_stats.outputs, 'Min', 'VECTOR'), node_subtract_02.inputs[1])

		#### Calculate normalised value
		links.new(node_subtract_01.outputs['Vector'], node_divide.inputs[0])
		links.new(node_subtract_02.outputs['Vector'], node_divide.inputs[1])

		#### Store normalised uv values
		links.new(node_divide.outputs['Vector'], node_store_named_attribute.inputs['Value'])
		node_store_named_attribute.inputs['Name'].default_value = 'geo_uv_texture'

		# Set locations 
		padding = 50
		group_in.location.x = 0
		group_in.location.y = 0

		self.place_node_alongside(node_mesh_to_points, group_in, padding)
		self.place_node_alongside(node_set_material, node_mesh_to_points, padding)
		self.place_node_alongside(node_store_named_attribute, node_set_material, padding)
		self.place_node_alongside(group_out, node_store_named_attribute, padding)

		node_self_object.location.x = -250.0
		node_self_object.location.y = -300.0
		self.place_node_alongside(node_self_object_info, node_self_object, padding, node_self_object.location.y)
		self.place_node_below(node_self_position, node_self_object_info, padding)
		self.place_node_alongside(node_self_add, node_self_object_info, padding, node_self_position.location.y)

		node_ref_object_info.location.x = -250.0
		node_ref_object_info.location.y = -800.0
		self.place_node_alongside(node_ref_stats, node_ref_object_info, padding, node_ref_object_info.location.y)
		self.place_node_below(node_ref_position, node_ref_object_info, padding)

		self.place_node_alongside(node_subtract_01, node_self_add, padding, node_self_add.location.y)
		self.place_node_below(node_subtract_02, node_subtract_01, padding)
		self.place_node_alongside(node_divide, node_subtract_01, padding, node_subtract_01.location.y)

		return node_group

	def scaled_dimension(self, las_file, targetCRS, fallbackCRS):

		target_crs = pyproj.CRS.from_string('EPSG:3857' if targetCRS == None else targetCRS)
		is_fallback = False

		try:
			source_crs = las_file.header.parse_crs()
		except pyproj.exceptions.CRSError:
			source_crs = None
		if source_crs == None:
			self.report({'ERROR'}, "Source CRS was not detected, assingning Fallback CRS " + fallbackCRS)
			source_crs = pyproj.CRS.from_string(fallbackCRS)
			is_fallback = True

		projecter = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

		xyz = las_file.xyz
		min_coords = np.array([las_file.header.x_min, las_file.header.y_min, las_file.header.z_min]).reshape(1, 3)
		max_coords = np.array([las_file.header.x_max, las_file.header.y_max, las_file.header.z_max]).reshape(1, 3)

		x, y = projecter.transform(xyz[:,0], xyz[:,1])
		min_x, min_y = projecter.transform(min_coords[:,0], min_coords[:,1])
		max_x, max_y = projecter.transform(max_coords[:,0], max_coords[:,1])

		xyz[:,0] = x
		xyz[:,1] = y
		min_coords[:,0] = min_x
		min_coords[:,1] = min_y
		max_coords[:,0] = max_x
		max_coords[:,1] = max_y
		xyz = np.array([item for item in xyz])
		coords = (min_coords[0], max_coords[0])
		return (xyz, coords, source_crs, is_fallback)


def register():
	try:
		bpy.utils.register_class(IMPORTLAZ_OT_georaster)
	except ValueError as e:
		log.warning('{} is already registered, now unregister and retry... '.format(IMPORTLAZ_OT_georaster))
		unregister()
		bpy.utils.register_class(IMPORTLAZ_OT_georaster)

def unregister():
	bpy.utils.unregister_class(IMPORTLAZ_OT_georaster)
