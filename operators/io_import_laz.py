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
import numpy as np#Ship with Blender since 2.70

import logging
log = logging.getLogger(__name__)

from ..geoscene import GeoScene, georefManagerLayout
from ..prefs import PredefCRS

from ..core.georaster import GeoRaster
from .utils import bpyGeoRaster, exportAsMesh
from .utils import placeObj, adjust3Dview, showTextures, addTexture, getBBOX
from .utils import rasterExtentToMesh, geoRastUVmap, setDisplacer
from .utils import getPointCloudMaterial, setPointColors, setPointCloudMaterial
from .utils.pointcloud_utils import POINTCLOUD_NODE_GROUP, COLOR_ATTRIBUTE
from .pointcloud_display import setColorBy
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

#Bundled as wheels (see blender_manifest.toml)
import laspy
import pyproj

PKG = __package__.rsplit('.', 1)[0]

def crs_to_scene_string(crs):
	'''Convert a pyproj CRS to the string representation expected by GeoScene.crs (AUTH:CODE, or proj4 as last resort)'''
	authority = crs.to_authority()
	if authority is not None:
		return ':'.join(authority)
	return crs.to_proj4()

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
	
	crsSource: EnumProperty(
		name = "CRS",
		description = "Where to read the point cloud Coordinate Reference System from",
		items = [
			('FILE', 'From file', "Use the CRS stored in the LAS header, or the CRS below when the file has none"),
			('CUSTOM', 'Custom', "Ignore the LAS header and always use the CRS below")
			],
		default = 'FILE'
		)

	customCRS: EnumProperty(
		name = "Point cloud CRS",
		description = "Coordinate Reference System of the point cloud, used when the file has none or when Custom is selected",
		items = listPredefCRS,
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
		
		# CRS handling: points are always reprojected to the scene CRS,
		# an ungeoreferenced scene adopts the point cloud CRS
		layout.prop(self, 'crsSource', expand=True)
		row = layout.row(align=True)
		split = row.split(factor=0.35, align=True)
		split.label(text='CRS if missing:' if self.crsSource == 'FILE' else 'CRS:')
		split.prop(self, "customCRS", text='')
		row.operator("bgis.add_predef_crs", text='', icon='ADD')

		if geoscn.isPartiallyGeoref:
			georefManagerLayout(self, context)
		else:
			layout.label(text="Scene will use the point cloud CRS", icon='INFO')

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

		# Handle clipping extent if requested (in scene CRS, like the reprojected points)
		subBox = None
		if self.clip:
			if not geoscn.isGeoref or len(self.objectsLst) == 0:
				self.report({'ERROR'}, "No georeferenced mesh available for clipping")
				return {'CANCELLED'}
			# Get chosen object for clipping extent
			clipObj = scn.objects[int(self.objectsLst)]
			subBox = getBBOX.fromObj(clipObj).toGeo(geoscn)

		new_objects_created = []

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

			source_crs, is_fallback = self.get_source_crs(las, name)

			# An ungeoreferenced scene adopts the CRS of the first point cloud, so no reprojection is needed
			if not geoscn.hasCRS:
				try:
					geoscn.crs = crs_to_scene_string(source_crs)
				except Exception as e:
					log.error("Cannot set scene crs", exc_info=True)
					self.report({'ERROR'}, "Cannot set scene crs, check logs for more infos")
					return {'CANCELLED'}

			# Get point cloud coordinates in scene CRS
			xyz = las.xyz.copy()
			scene_crs = pyproj.CRS.from_user_input(geoscn.crs)
			if not source_crs.equals(scene_crs, ignore_axis_order=True):
				# Vectorized reprojection (much faster than a Reproj.pt loop)
				transformer = pyproj.Transformer.from_crs(source_crs, scene_crs, always_xy=True)
				xyz[:, 0], xyz[:, 1] = transformer.transform(xyz[:, 0], xyz[:, 1])

			# Apply clipping if requested
			mask = None
			if subBox:
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
				'las': las,
				'mask': mask # clipping mask, to keep per point LAS data aligned with xyz
			})

		# Set scene origin at the center of all point clouds if not already georeferenced
		if not geoscn.isGeoref and new_objects_created:
			all_points = np.vstack([obj_data['xyz'] for obj_data in new_objects_created])
			center_x = (np.min(all_points[:, 0]) + np.max(all_points[:, 0])) / 2
			center_y = (np.min(all_points[:, 1]) + np.max(all_points[:, 1])) / 2
			geoscn.setOriginPrj(center_x, center_y)

		if geoscn.isGeoref:
			dx, dy = geoscn.getOriginPrj()

		# Now create the actual Blender objects with proper positioning
		final_objects = []
		for obj_data in new_objects_created:
			xyz = obj_data['xyz']
			name = obj_data['name']

			# Offset by scene origin
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
				self.add_point_attributes(pc, obj_data['las'], obj_data['mask'])
			self.add_point_colors(pc, obj_data['las'], obj_data['mask'])

			# Create and place object - using BlenderGIS placeObj utility
			obj = placeObj(pc, name)
			
			# Store metadata on object
			obj['source_crs'] = obj_data['source_crs'].to_string() if hasattr(obj_data['source_crs'], 'to_string') else str(obj_data['source_crs'])
			obj['is_fallback_crs'] = obj_data['is_fallback']
			obj['import_scale'] = self.import_scale
			obj['point_count'] = xyz.shape[0]

			# Apply geometry nodes for point visualization
			self.assign_geometry_node(obj, context)
			# Show LAS RGB when available, elevation otherwise (changeable in the Point Cloud panel)
			setColorBy(obj, 'RGB' if COLOR_ATTRIBUTE in pc.color_attributes else 'ELEVATION')
			
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

	def get_source_crs(self, las_file, name):
		"""Return the horizontal CRS of the point cloud and whether it comes from the user choice instead of the file"""
		source_crs = None
		if self.crsSource == 'FILE':
			try:
				source_crs = las_file.header.parse_crs()
			except (pyproj.exceptions.CRSError, AttributeError):
				log.warning("Cannot parse CRS from LAS header", exc_info=True)
			if source_crs is None:
				self.report({'WARNING'}, f"No CRS found in {name}, using {self.customCRS}")
		is_custom = source_crs is None
		if is_custom:
			source_crs = pyproj.CRS.from_user_input(self.customCRS)
		# Keep the horizontal part of compound CRS (horizontal + vertical), z values are left untouched
		if source_crs.is_compound:
			source_crs = next((sub for sub in source_crs.sub_crs_list if sub.is_projected or sub.is_geographic), source_crs)
		return source_crs, is_custom

	# Bit-field sub-views laspy unpacks from packed bytes — usually redundant
	# (the underlying packed dim is also exposed) and expensive to materialize.
	_BITFIELD_SUBFIELDS = frozenset({
		'return_number', 'number_of_returns',
		'scan_direction_flag', 'edge_of_flight_line',
		'classification_flags', 'scanner_channel',
		'synthetic', 'key_point', 'withheld', 'overlap',
	})

	def add_point_attributes(self, mesh, las_file, mask=None):
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
				if mask is not None:
					attr_data = attr_data[mask]
				mesh.attributes[attr_name].data.foreach_set("value", attr_data)
			except Exception as e:
				log.warning(f"Could not add attribute {attr_name}: {e}")

	def add_point_colors(self, mesh, las_file, mask=None):
		"""Store LAS RGB as the per point color attribute displayed by the point cloud material"""
		if not {'red', 'green', 'blue'} <= set(las_file.point_format.dimension_names):
			return
		rgb = np.column_stack([np.asarray(las_file.red), np.asarray(las_file.green), np.asarray(las_file.blue)]).astype(np.float32)
		if mask is not None:
			rgb = rgb[mask]
		# LAS stores 16 bits colors, but many writers put 8 bits values in it
		rgb /= 255.0 if rgb.max() <= 255 else 65535.0
		rgba = np.ones((rgb.shape[0], 4), dtype=np.float32)
		rgba[:, :3] = rgb
		setPointColors(mesh, rgba)

	def assign_geometry_node(self, obj, context):
		"""Assign geometry node group for point cloud visualization"""
		node_group_name = POINTCLOUD_NODE_GROUP
		if node_group_name in bpy.data.node_groups:
			node_group = bpy.data.node_groups[node_group_name]
		else:
			node_group = self.new_pointcloud_geometry_node_group(context, getPointCloudMaterial())

		modifier = obj.modifiers.new(node_group_name, 'NODES')
		modifier.node_group = node_group
		# Shared template group: make sure it shows the point colors (heals groups edited by older versions)
		setPointCloudMaterial(obj, getPointCloudMaterial(), singleUser=False)
		# Set point size from user preference
		if hasattr(modifier, "properties"): # Blender 5.x: inputs are no longer IDProperties of the modifier
			pointSize = getattr(modifier.properties.inputs, "Socket_2", None)
			if pointSize is not None:
				pointSize.value = self.point_size
		elif "Socket_2" in modifier:
			modifier["Socket_2"] = self.point_size

	# ... [Rest of the geometry node methods remain the same as they handle visualization, 
	# not the core georeferencing logic] ...

	def place_node_alongside(self, node, other, padding, y = None):
		node.location.x = other.location.x + other.width + padding
		node.location.y = 0.0 if y == None else other.location.y
	
	def place_node_below(self, node, other, padding):
		node.location.x = other.location.x
		node.location.y = other.location.y - other.height - padding

	def node_for_type(self, sources, source_name, type_name):
		return [source for source in sources if source.type == type_name and source.name == source_name][0]

	def new_pointcloud_geometry_node_group(self, context, material_pointcloud):
		''' Create a new empty node group that can be used
			in a GeometryNodes modifier.
		'''
		node_group = bpy.data.node_groups.new(POINTCLOUD_NODE_GROUP, 'GeometryNodeTree')
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