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
from ..core.utils import BBOX

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
	import pyproj
except ImportError:
	env = os.environ.copy()
	for dep_name in ("laspy[lazrs]", "pyproj"):
		res = subprocess.run( [sys.executable, "-m", "pip", "install", dep_name], env=env)
	import laspy
	import pyproj

PKG, SUBPKG = __package__.split('.', maxsplit=1)

class IMPORTLAZ_OT_georaster(Operator, ImportHelper):
	"""Import georeferenced LAZ/LAS point cloud"""
	bl_idname = "importgis.laz"  # important since its how bpy.ops.importgis.laz is constructed (allows calling operator from python console or another script)
	#bl_idname rules: must contain one '.' (dot) charactere, no capital letters, no reserved words (like 'import')
	bl_description = 'Import LAZ/LAS point cloud'
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
			default="*.laz;*.las",
			options={'HIDDEN'},
			)

	# CRS definition
	def listPredefCRS(self, context):
		return PredefCRS.getEnumItems()

	files: CollectionProperty(
		type = bpy.types.OperatorFileListElement,
		options = {'HIDDEN', 'SKIP_SAVE'}
	)
	
	pointCRS: EnumProperty(
		name = "Point Cloud CRS",
		description = "Choose a Coordinate Reference System",
		items = listPredefCRS,
		)
	
	fallbackCRS: EnumProperty(
		name = "Fallback CRS",
		description = "Choose a Coordinate Reference System when LIDAR data doesn't contain CRS metadata",
		items = listPredefCRS,
		)
	
	use_fallback: BoolProperty(
		name="Use Fallback",
		description="Skip parsing CRS from LAS header and use fallback CRS directly",
		default=False
		)
	
	import_scale: FloatProperty(
		name = "Import Scale",
		description = "Scale factor for imported point cloud",
		default=1.0,
		min=0.001,
		max=1000.0
	)

	import_attributes: BoolProperty(
		name="Import LAS attributes",
		description="Add LAS dimensions (intensity, classification, RGB, etc.) as Blender point attributes. Disable for fastest import",
		default=True,
	)

	reprojection: BoolProperty(
			name="Specify point cloud CRS",
			description="Specify point cloud CRS if it's different from scene CRS",
			default=False )
	
	# Point cloud specific options
	point_size: FloatProperty(
		name = "Point Size",
		description = "Size of individual points in the visualization",
		default=0.5,
		min=0.01,
		max=10.0
	)
	
	clip: BoolProperty(
		name="Clip to working extent",
		description="Use the reference bounding box to clip the point cloud",
		default=False
	)
	
	objectsLst: EnumProperty(attr="obj_list", name="Objects", description="Choose object to clip against", items=listObjects)

	def draw(self, context):
		#Function used by blender to draw the panel.
		layout = self.layout
		scn = bpy.context.scene
		geoscn = GeoScene(scn)
		
		# Point cloud specific settings
		layout.prop(self, 'point_size')
		layout.prop(self, 'import_scale')
		layout.prop(self, 'import_attributes')
		
		# Clipping options
		layout.prop(self, 'clip')
		if self.clip:
			if geoscn.isGeoref and len(self.objectsLst) > 0:
				layout.prop(self, 'objectsLst')
			else:
				layout.label(text="No georeferenced mesh available for clipping")
		
		# CRS handling
		if geoscn.isPartiallyGeoref:
			layout.prop(self, 'reprojection')
			if self.reprojection:
				self.crsInputLayout(context)
			georefManagerLayout(self, context)
		else:
			self.crsInputLayout(context)
		
		self.fallbackCRSInputLayout(context)

	def crsInputLayout(self, context):
		layout = self.layout
		row = layout.row(align=True)
		split = row.split(factor=0.35, align=True)
		split.label(text='Point Cloud CRS:')
		split.prop(self, "pointCRS", text='')
		row.operator("bgis.add_predef_crs", text='', icon='ADD')

	def fallbackCRSInputLayout(self, context):
		layout = self.layout
		row = layout.row(align=True)
		split = row.split(factor=0.35, align=True)
		split.label(text='Fallback CRS:')
		split.prop(self, "fallbackCRS", text='')
		row.operator("bgis.add_predef_crs", text='', icon='ADD')
		layout.prop(self, "use_fallback")

	@classmethod
	def poll(cls, context):
		return context.mode == 'OBJECT'

	def execute(self, context):
		prefs = context.preferences.addons[PKG].preferences

		bpy.ops.object.select_all(action='DESELECT')
		
		# Get scene and georef data - following IMPORTGIS_OT_georaster pattern
		scn = bpy.context.scene
		geoscn = GeoScene(scn)
		if geoscn.isBroken:
			self.report({'ERROR'}, "Scene georef is broken, please fix it beforehand")
			return {'CANCELLED'}

		scale = geoscn.scale

		# Handle CRS setup - following IMPORTGIS_OT_georaster pattern
		if geoscn.isGeoref:
			dx, dy = geoscn.getOriginPrj()
			if self.reprojection:
				pointCRS = self.pointCRS
			else:
				pointCRS = geoscn.crs
		else: #if not geoscn.hasCRS
			pointCRS = self.pointCRS
			try:
				geoscn.crs = pointCRS
			except Exception as e:
				log.error("Cannot set scene crs", exc_info=True)
				self.report({'ERROR'}, "Cannot set scene crs, check logs for more infos")
				return {'CANCELLED'}

		# Point cloud reprojection setup - following IMPORTGIS_OT_georaster pattern
		if geoscn.crs != pointCRS:
			rprj = True
			rprjToPointCloud = Reproj(geoscn.crs, pointCRS)
			rprjToScene = Reproj(pointCRS, geoscn.crs)
			# Vectorized transformer for per-vertex reprojection (much faster than Reproj.pt loop)
			pyprojToScene = pyproj.Transformer.from_crs(
				pyproj.CRS.from_string(pointCRS),
				pyproj.CRS.from_string(geoscn.crs),
				always_xy=True,
			)
		else:
			rprj = False
			rprjToPointCloud = None
			rprjToScene = None
			pyprojToScene = None

		# Handle clipping extent if requested
		subBox = None
		if self.clip:
			if not geoscn.isGeoref or len(self.objectsLst) == 0:
				self.report({'ERROR'}, "No georeferenced mesh available for clipping")
				return {'CANCELLED'}
			# Get chosen object for clipping extent
			clipObj = scn.objects[int(self.objectsLst)]
			subBox = getBBOX.fromObj(clipObj).toGeo(geoscn)
			if rprj:
				subBox = rprjToPointCloud.bbox(subBox)

		new_objects_created = []
		all_original_coords = []  # Store original coordinates for global centroid calculation

		# Process each LAZ/LAS file
		for f in self.files:
			filePath = os.path.join(os.path.dirname(self.filepath), f.name)
			name = os.path.basename(filePath)[:-4]

			try:
				las = laspy.read(filePath)
			except IOError as e:
				log.error("Unable to open LAZ/LAS file", exc_info=True)
				self.report({'ERROR'}, f"Unable to open {name}, check logs for more infos")
				continue

			# Get point cloud coordinates and handle CRS
			xyz, source_crs, is_fallback = self.get_transformed_coordinates(las, pointCRS, self.fallbackCRS, self.use_fallback)
			
			if xyz is None:
				continue

			# Store original coordinates for global centroid calculation
			if not geoscn.isGeoref:
				all_original_coords.append(xyz.copy())

			# Apply clipping if requested
			if subBox:
				# Convert subBox to point cloud coordinates for clipping
				mask = ((xyz[:, 0] >= subBox.xmin) & (xyz[:, 0] <= subBox.xmax) & 
						(xyz[:, 1] >= subBox.ymin) & (xyz[:, 1] <= subBox.ymax))
				if not np.any(mask):
					self.report({'WARNING'}, f"No points in clipping extent for {name}")
					continue
				xyz = xyz[mask]

			# Store the processed coordinates and metadata for later positioning
			new_objects_created.append({
				'name': name,
				'xyz': xyz,
				'source_crs': source_crs,
				'is_fallback': is_fallback,
				'las': las
			})

		# Calculate global scene origin if not already georeferenced
		if not geoscn.isGeoref and all_original_coords:
			# Calculate global centroid from all original point clouds
			all_points = np.vstack(all_original_coords)
			center_x = (np.min(all_points[:, 0]) + np.max(all_points[:, 0])) / 2
			center_y = (np.min(all_points[:, 1]) + np.max(all_points[:, 1])) / 2
			if rprj:
				center_x, center_y = rprjToScene.pt(center_x, center_y)
			dx, dy = center_x, center_y
			geoscn.setOriginPrj(dx, dy)

		# Now create the actual Blender objects with proper positioning
		final_objects = []
		for obj_data in new_objects_created:
			xyz = obj_data['xyz']
			name = obj_data['name']
			
			# Transform to scene coordinates if needed (vectorized via pyproj)
			if rprj:
				tx, ty = pyprojToScene.transform(xyz[:, 0], xyz[:, 1])
				xyz[:, 0] = tx
				xyz[:, 1] = ty

			# Offset by scene origin (now that we have the global origin)
			if geoscn.isGeoref:
				xyz[:, 0] -= dx
				xyz[:, 1] -= dy

			# Apply import scale
			xyz *= self.import_scale

			# Create mesh from point cloud (foreach_set on a flat numpy buffer is much faster than from_pydata)
			pc = bpy.data.meshes.new(name)
			n_points = xyz.shape[0]
			pc.vertices.add(n_points)
			pc.vertices.foreach_set("co", np.ascontiguousarray(xyz, dtype=np.float32).ravel())
			pc.update()

			if self.import_attributes:
				self.add_point_attributes(pc, obj_data['las'], xyz.shape[0])

			# Create and place object - using BlenderGIS placeObj utility
			obj = placeObj(pc, name)
			
			# Store metadata on object
			obj['source_crs'] = obj_data['source_crs'].to_string() if hasattr(obj_data['source_crs'], 'to_string') else str(obj_data['source_crs'])
			obj['is_fallback_crs'] = obj_data['is_fallback']
			obj['import_scale'] = self.import_scale
			obj['point_count'] = xyz.shape[0]

			# Apply geometry nodes for point visualization
			self.assign_geometry_node(obj, context)
			
			final_objects.append(obj)

		if not final_objects:
			self.report({'ERROR'}, "No point clouds were successfully imported")
		# Adjust 3D view if preference is set - following IMPORTGIS_OT_georaster pattern
		if prefs.adjust3Dview and final_objects:
			# Calculate combined bounding box manually
			if len(final_objects) == 1:
				bb = getBBOX.fromObj(final_objects[0])
			else:
				# For multiple objects, calculate encompassing bounds
				all_coords = []
				for obj in final_objects:
					bbox = getBBOX.fromObj(obj)
					all_coords.extend([
						(bbox.xmin, bbox.ymin, bbox.zmin),
						(bbox.xmax, bbox.ymax, bbox.zmax)
					])
				
				if all_coords:
					min_x = min(coord[0] for coord in all_coords)
					max_x = max(coord[0] for coord in all_coords)
					min_y = min(coord[1] for coord in all_coords)
					max_y = max(coord[1] for coord in all_coords)
					min_z = min(coord[2] for coord in all_coords)
					max_z = max(coord[2] for coord in all_coords)
					
					# Create a combined bounding box using the same class as getBBOX returns
					bb = BBOX(xmin=min_x, xmax=max_x, ymin=min_y, ymax=max_y, zmin=min_z, zmax=max_z)
				else:
					bb = getBBOX.fromObj(final_objects[0])
			
			adjust3Dview(context, bb)

		self.report({'INFO'}, f"Successfully imported {len(final_objects)} point cloud(s)")
		return {'FINISHED'}

	def get_transformed_coordinates(self, las_file, target_crs_str, fallback_crs_str, use_fallback=False):
		"""Extract and transform coordinates from LAS file, handling CRS detection"""
		if use_fallback:
			# Skip header parsing and use fallback CRS directly
			source_crs = pyproj.CRS.from_string(fallback_crs_str)
			is_fallback = True
			self.report({'INFO'}, f"Using fallback CRS as requested: {fallback_crs_str}")
		else:
			source_crs = None
			try:
				# Try to get CRS from LAS header
				source_crs = las_file.header.parse_crs()
			except (pyproj.exceptions.CRSError, AttributeError):
				source_crs = None
			if source_crs:
				is_fallback = False
			else:
				# Header had no CRS (parse_crs returned None) or parsing failed — use fallback
				source_crs = pyproj.CRS.from_string(fallback_crs_str)
				is_fallback = True
				self.report({'WARNING'}, f"Could not detect CRS from file, using fallback: {fallback_crs_str}")

		if not source_crs:
			self.report({'ERROR'}, "Could not determine coordinate reference system")
			return None, None, None

		# Setup coordinate transformation if needed
		target_crs = pyproj.CRS.from_string(target_crs_str)
		
		xyz = las_file.xyz.copy()
		
		if source_crs != target_crs:
			transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
			x_trans, y_trans = transformer.transform(xyz[:, 0], xyz[:, 1])
			xyz[:, 0] = x_trans
			xyz[:, 1] = y_trans

		return xyz, source_crs, is_fallback

	# Bit-field sub-views laspy unpacks from packed bytes — usually redundant
	# (the underlying packed dim is also exposed) and expensive to materialize.
	_BITFIELD_SUBFIELDS = frozenset({
		'return_number', 'number_of_returns',
		'scan_direction_flag', 'edge_of_flight_line',
		'classification_flags', 'scanner_channel',
		'synthetic', 'key_point', 'withheld', 'overlap',
	})

	def add_point_attributes(self, mesh, las_file, point_count):
		"""Add LAS point attributes to Blender mesh"""

		for attr_name in las_file.point_format.dimension_names:
			if attr_name in ('X', 'Y', 'Z'):
				continue
			if attr_name in self._BITFIELD_SUBFIELDS:
				continue  # Skip coordinates
			dim_info = las_file.point_format.dimension_by_name(attr_name)
			try:
				dtype = dim_info.dtype
				attr_type = 'INT' if np.issubdtype(dtype, np.integer) else 'FLOAT'
				domain = 'POINT'
				mesh.attributes.new(name=attr_name, type=attr_type, domain=domain)
				attr_data = np.asarray(getattr(las_file, attr_name))
				# Handle potential clipping by only taking the first point_count values
				if len(attr_data) > point_count:
					attr_data = attr_data[:point_count]
				mesh.attributes[attr_name].data.foreach_set("value", attr_data)
			except Exception as e:
				log.warning(f"Could not add attribute {attr_name}: {e}")

	def assign_geometry_node(self, obj, context):
		"""Assign geometry node group for point cloud visualization"""
		node_group_name = "Point Cloud Visualisation Node"
		if node_group_name in bpy.data.node_groups:
			node_group = bpy.data.node_groups[node_group_name]
		else:
			material_pointcloud = self.new_pointcloud_material()
			node_group = self.new_pointcloud_geometry_node_group(context, material_pointcloud)
		
		modifier = obj.modifiers.new(node_group_name, 'NODES')
		modifier.node_group = node_group
		# Set point size from user preference
		if "Socket_2" in modifier:
			modifier["Socket_2"] = self.point_size

	# ... [Rest of the geometry node methods remain the same as they handle visualization, 
	# not the core georeferencing logic] ...

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

		return mat

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

		# Set properties
		self.create_interface_socket(node_group, 'Geometry', 'OUTPUT', 'NodeSocketGeometry')
		self.create_interface_socket(node_group, 'Geometry', 'INPUT', 'NodeSocketGeometry')
		self.create_interface_socket(node_group, 'Radius', 'INPUT', 'NodeSocketFloat')
		
		node_mesh_to_points.inputs['Radius'].default_value = self.point_size

		# Set links
		links.new(group_in.outputs['Geometry'], node_mesh_to_points.inputs['Mesh'])
		links.new(group_in.outputs['Radius'], node_mesh_to_points.inputs['Radius'])
		links.new(node_mesh_to_points.outputs['Points'], node_set_material.inputs['Geometry'])
		node_set_material.inputs['Material'].default_value = material_pointcloud
		links.new(node_set_material.outputs['Geometry'], group_out.inputs['Geometry'])

		# Set locations 
		padding = 50
		group_in.location.x = 0
		group_in.location.y = 0

		self.place_node_alongside(node_mesh_to_points, group_in, padding)
		self.place_node_alongside(node_set_material, node_mesh_to_points, padding)
		self.place_node_alongside(group_out, node_set_material, padding)

		return node_group

	def create_interface_socket(self, node_group, socket_name, in_out, socket_type):
		group_name = 'Group Input' if in_out == 'INPUT' else 'Group Output'
		node_name = 'Node' + group_name.replace(' ', '')
		if not group_name in node_group.nodes:
			node = node_group.nodes.new(node_name)
		if hasattr(node_group, 'interface'):
			socket_type = 'NodeSocketInt' if socket_type == 'NodeSocketIntUnsigned' else socket_type
			node_group.interface.new_socket(socket_name, in_out=in_out, socket_type=socket_type)
			return node_group.nodes[group_name]
		else:
			node = node_group.nodes.new(node_name)
			if in_out == 'INPUT':
				node_group.inputs.new(socket_type, socket_name)
			else:
				node_group.outputs.new(socket_type, socket_name)
		return node


def register():
	try:
		bpy.utils.register_class(IMPORTLAZ_OT_georaster)
	except ValueError as e:
		log.warning('{} is already registered, now unregister and retry... '.format(IMPORTLAZ_OT_georaster))
		unregister()
		bpy.utils.register_class(IMPORTLAZ_OT_georaster)

def unregister():
	bpy.utils.unregister_class(IMPORTLAZ_OT_georaster)