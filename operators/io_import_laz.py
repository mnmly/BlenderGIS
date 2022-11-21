
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
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty, CollectionProperty
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
	reprojection: BoolProperty(
			name="Specifiy raster CRS",
			description="Specifiy raster CRS if it's different from scene CRS",
			default=False )

	# List of operator properties, the attributes will be assigned
	# to the class instance from the operator settings before calling.
	importMode: EnumProperty(
			name="Mode",
			description="Select import mode",
			items=[ ('PLANE', 'Basemap on new plane', "Place raster texture on new plane mesh"),
			('BKG', 'Basemap as background', "Place raster as background image"),
			('MESH', 'Basemap on mesh', "UV map raster on an existing mesh"),
			('DEM', 'DEM as displacement texture', "Use DEM raster as height texture to wrap a base mesh"),
			('DEM_RAW', 'DEM raw data build [slow]', "Import a DEM as pixels points cloud with building faces. Do not use with huge dataset.")]
			)
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
		layout.prop(self, 'importMode')
		scn = bpy.context.scene
		geoscn = GeoScene(scn)
		#
		if self.importMode == 'PLANE':
			pass
		#
		if self.importMode == 'BKG':
			pass
		#
		if self.importMode == 'MESH':
			if geoscn.isGeoref and len(self.objectsLst) > 0:
				layout.prop(self, 'objectsLst')
			else:
				layout.label(text="There isn't georef mesh to UVmap on")
		#
		if self.importMode == 'DEM':
			layout.prop(self, 'demOnMesh')
			if self.demOnMesh:
				if geoscn.isGeoref and len(self.objectsLst) > 0:
					layout.prop(self, 'objectsLst')
					layout.prop(self, 'clip')
				else:
					layout.label(text="There isn't georef mesh to apply on")
			layout.prop(self, 'subdivision')
			layout.prop(self, 'demInterpolation')
			if self.subdivision == 'mesh':
				layout.prop(self, 'step')
			layout.prop(self, 'fillNodata')
		#
		if self.importMode == 'DEM_RAW':
			layout.prop(self, 'buildFaces')
			layout.prop(self, 'step')
			layout.prop(self, 'clip')
			if self.clip:
				if geoscn.isGeoref and len(self.objectsLst) > 0:
					layout.prop(self, 'objectsLst')
				else:
					layout.label(text="There isn't georef mesh to refer")
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
			verts = self.scaled_dimension(las, geoscn.crs if geoscn.hasCRS else None, self.fallbackCRS)
			pc.from_pydata(verts, [], [])
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
			if geoscn.crsx != None:
				obj.location.x = -geoscn.crsx
				obj.location.y = -geoscn.crsy
		return {'FINISHED'}

	def scaled_dimension(self, las_file, targetCRS, fallbackCRS):
		target_crs = pyproj.CRS.from_string('EPSG:3857' if targetCRS == None else targetCRS)
		source_crs = las_file.header.parse_crs()
		if source_crs == None:
			self.report({'ERROR'}, "Source CRS was not detected, assingning Fallback CRS " + fallbackCRS)
			source_crs = pyproj.CRS.from_string(fallbackCRS)
		projecter = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
		xyz = las_file.xyz
		x, y = projecter.transform(xyz[:,0], xyz[:,1])	
		xyz[:,0] = x
		xyz[:,1] = y
		return [item for item in xyz]


def register():
	try:
		bpy.utils.register_class(IMPORTLAZ_OT_georaster)
	except ValueError as e:
		log.warning('{} is already registered, now unregister and retry... '.format(IMPORTLAZ_OT_georaster))
		unregister()
		bpy.utils.register_class(IMPORTLAZ_OT_georaster)

def unregister():
	bpy.utils.unregister_class(IMPORTLAZ_OT_georaster)
