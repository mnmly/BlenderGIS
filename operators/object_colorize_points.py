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

import os
import logging
log = logging.getLogger(__name__)

import bpy
import numpy as np
import pyproj #bundled as wheel (see blender_manifest.toml)

from bpy.props import StringProperty, EnumProperty
from bpy.types import Operator

from ..geoscene import GeoScene
from ..core.georaster import GeoRaster
from ..prefs import PredefCRS
from .utils import bpyGeoRaster, newRasterPointCloudMaterial, setPointColors, getPointColors
from .utils.pointcloud_utils import UV_ATTRIBUTE, POINTCLOUD_RASTER_MATERIAL

#custom properties set by the georaster importer on the objects it creates
RASTER_PATH_PROP = 'bgis_raster_path'
RASTER_CRS_PROP = 'bgis_raster_crs'
from .pointcloud_display import setColorBy


def pointsGeoCoords(obj, geoscn):
	'''World xy coordinates of the mesh vertices, in scene CRS space'''
	mesh = obj.data
	co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
	mesh.vertices.foreach_get('co', co)
	co = co.reshape(-1, 3).astype(np.float64)
	mat = np.array(obj.matrix_world, dtype=np.float64)
	world = co @ mat[:3, :3].T + mat[:3, 3]
	#point clouds imported by BlenderGIS can be scaled at import, other meshes are in scene units
	scale = obj.get('import_scale', 1.0)
	dx, dy = geoscn.getOriginPrj()
	return world[:, 0] / scale + dx, world[:, 1] / scale + dy

def geoToPx(georef, x, y):
	'''Vectorized GeoRef.pxFromGeo: continuous pixel coordinates from the raster top left corner'''
	pxSizex, pxSizey = georef.pxSize
	rotx, roty = georef.rotation
	offx = georef.origin.x - abs(pxSizex / 2)
	offy = georef.origin.y + abs(pxSizey / 2)
	det = pxSizex * pxSizey - rotx * roty
	xPx = (pxSizey * x - rotx * y + rotx * offy - pxSizey * offx) / det
	yPx = (-roty * x + pxSizex * y + roty * offx - pxSizex * offy) / det
	return xPx, yPx

def linearToSrgb(c):
	return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0.0031308, None), 1 / 2.4) - 0.055)


def rasterImageOf(obj):
	"""Georeferenced raster displayed by a mesh (e.g. a BlenderGIS raster plane): its file backed image texture, or None"""
	if obj.type != 'MESH':
		return None
	for mat in obj.data.materials:
		if mat is None or mat.node_tree is None:
			continue
		for node in mat.node_tree.nodes:
			if node.type == 'TEX_IMAGE' and node.image is not None and node.image.source == 'FILE':
				if os.path.isfile(bpy.path.abspath(node.image.filepath, library=node.image.library)):
					return node.image
	return None

def rasterPath(obj, img):
	"""File holding the raster georeferencing: the path stored at import, or the image file"""
	path = obj.get(RASTER_PATH_PROP)
	if path and os.path.isfile(path):
		return path
	return bpy.path.abspath(img.filepath, library=img.library)

#Blender needs the strings of dynamic enum items to stay referenced
_rasterItems = []

def listSceneRasters(self, context):
	_rasterItems[:] = [(obj.name, obj.name, "Raster displayed by " + obj.name, 'IMAGE_DATA', i)
		for i, obj in enumerate(o for o in context.scene.objects if rasterImageOf(o) is not None)]
	return _rasterItems or [('NONE', 'No raster in the scene', "Import a georeferenced raster first", 'ERROR', 0)]


class OBJECT_OT_colorize_points(Operator):
	"""Color point clouds (or any mesh vertices) from a georeferenced raster, like an orthophoto, projected from above"""
	bl_idname = "object.colorize_points"
	bl_description = 'Color selected point clouds from a georeferenced raster, in the scene or from a file'
	bl_label = "Colorize points from raster"
	bl_options = {"UNDO"}

	source: EnumProperty(
		name = "Source",
		items = [
			('SCENE', 'Scene raster', "A georeferenced raster already in the scene, e.g. imported with BlenderGIS or from a QGIS project"),
			('FILE', 'File', "A georeferenced raster file: GeoTIFF, or an image with a world file")
			],
		default = 'FILE'
		)

	rasterObject: EnumProperty(
		name = "Raster",
		description = "Raster object of the scene whose image colors the points",
		items = listSceneRasters,
		)

	filepath: StringProperty(
		name = "File",
		description = "Georeferenced raster: GeoTIFF, or an image (jpg, png, tif...) with a world file",
		subtype = 'FILE_PATH',
		)

	def listRasterCRS(self, context):
		return [('SCENE', 'Same as scene', "The raster uses the scene CRS")] + PredefCRS.getEnumItems()

	rasterCRS: EnumProperty(
		name = "Raster CRS",
		description = "Coordinate Reference System of the raster",
		items = listRasterCRS,
		)

	mode: EnumProperty(
		name = "Mode",
		items = [
			('LIVE', 'Live', "Project the raster through the point cloud material: fast, no pixels copied, the raster can be changed later"),
			('BAKE', 'Bake', "Copy the raster colors into the point color attribute: kept on export, the raster is not needed anymore")
			],
		default = 'LIVE'
		)

	@classmethod
	def poll(cls, context):
		return context.mode == 'OBJECT' and any(obj.type == 'MESH' for obj in context.selected_objects)

	def invoke(self, context, event):
		#prefer a raster already in the scene, like a layer imported from a QGIS project
		hasSceneRaster = listSceneRasters(self, context)[0][0] != 'NONE'
		self.source = 'SCENE' if hasSceneRaster else 'FILE'
		return context.window_manager.invoke_props_dialog(self, width=420)

	def sceneRaster(self, context):
		obj = context.scene.objects.get(self.rasterObject)
		return obj, (rasterImageOf(obj) if obj is not None else None)

	def draw(self, context):
		layout = self.layout
		layout.prop(self, 'source', expand=True)
		storedCRS = None
		if self.source == 'SCENE':
			layout.prop(self, 'rasterObject')
			obj, img = self.sceneRaster(context)
			storedCRS = obj.get(RASTER_CRS_PROP) if obj is not None else None
		else:
			layout.prop(self, 'filepath')
		if storedCRS:
			layout.label(text=f"Raster CRS: {storedCRS}", icon='INFO')
		else:
			layout.prop(self, 'rasterCRS')
		layout.prop(self, 'mode', expand=True)

	def execute(self, context):
		geoscn = GeoScene(context.scene)
		if not geoscn.isGeoref:
			self.report({'ERROR'}, "Scene is not georeferenced, import the point cloud first")
			return {'CANCELLED'}

		rasterCRS = self.rasterCRS
		fileRaster = rasterObj = None
		try:
			if self.source == 'SCENE':
				rasterObj, img = self.sceneRaster(context)
				if img is None:
					self.report({'ERROR'}, "Choose a georeferenced raster of the scene")
					return {'CANCELLED'}
				#georeferencing comes from the raster file, colors from the image already loaded in Blender
				georef = GeoRaster(rasterPath(rasterObj, img)).georef
				rasterCRS = rasterObj.get(RASTER_CRS_PROP, rasterCRS)
			else:
				fileRaster = bpyGeoRaster(bpy.path.abspath(self.filepath))
				img, georef = fileRaster.bpyImg, fileRaster.georef
		except IOError as e:
			log.error("Unable to open raster", exc_info=True)
			self.report({'ERROR'}, f"Unable to read the georeferenced raster: {e}")
			return {'CANCELLED'}
		w, h = georef.rSize
		if tuple(img.size) != (w, h):
			self.report({'ERROR'}, f"Image size {tuple(img.size)} does not match its georeferencing {(w, h)}")
			return {'CANCELLED'}

		if rasterCRS != 'SCENE' and rasterCRS != geoscn.crs:
			toRaster = pyproj.Transformer.from_crs(pyproj.CRS.from_user_input(geoscn.crs), pyproj.CRS.from_user_input(rasterCRS), always_xy=True)
		else:
			toRaster = None

		pixels = None
		if self.mode == 'BAKE':
			pixels = np.empty(w * h * 4, dtype=np.float32)
			img.pixels.foreach_get(pixels)
			pixels = pixels.reshape(h, w, 4) #Blender stores rows from bottom to top
			if img.is_float:
				pixels[..., :3] = linearToSrgb(pixels[..., :3])
		else:
			mat = bpy.data.materials.get(POINTCLOUD_RASTER_MATERIAL + '.' + img.name) or newRasterPointCloudMaterial(img)

		nPoints = nInside = 0
		for obj in [obj for obj in context.selected_objects if obj.type == 'MESH' and obj != rasterObj]:
			mesh = obj.data
			x, y = pointsGeoCoords(obj, geoscn)
			if toRaster is not None:
				x, y = toRaster.transform(x, y)
			xPx, yPx = geoToPx(georef, x, y)
			inside = (xPx >= 0) & (xPx < w) & (yPx >= 0) & (yPx < h)
			nPoints += len(inside)
			nInside += int(inside.sum())

			if self.mode == 'LIVE':
				uv = np.column_stack([xPx / w, 1 - yPx / h]).astype(np.float32)
				if UV_ATTRIBUTE in mesh.attributes:
					mesh.attributes.remove(mesh.attributes[UV_ATTRIBUTE])
				mesh.attributes.new(UV_ATTRIBUTE, 'FLOAT2', 'POINT').data.foreach_set('vector', uv.ravel())
				obj.bgis_pointcloud.rasterMaterial = mat
				setColorBy(obj, 'RASTER')
			else:
				rgba = getPointColors(mesh)
				if rgba is None:
					rgba = np.zeros((len(mesh.vertices), 4), dtype=np.float32) #alpha 0 shows the grey fallback
				cols = np.floor(xPx[inside]).astype(np.int64)
				rows = h - 1 - np.floor(yPx[inside]).astype(np.int64)
				rgba[inside] = pixels[rows, cols]
				setPointColors(mesh, rgba)
				setColorBy(obj, 'RGB')
			mesh.update()

		if fileRaster is not None and self.mode == 'BAKE' and img.users == 0:
			fileRaster.unload()

		if nInside == 0:
			self.report({'WARNING'}, "The raster does not overlap the selected points, check the raster CRS")
		else:
			self.report({'INFO'}, f"Colored {nInside} of {nPoints} points ({100 * nInside / nPoints:.1f}%)")
		return {'FINISHED'}


def register():
	try:
		bpy.utils.register_class(OBJECT_OT_colorize_points)
	except ValueError as e:
		log.warning('{} is already registered, now unregister and retry... '.format(OBJECT_OT_colorize_points))
		unregister()
		bpy.utils.register_class(OBJECT_OT_colorize_points)

def unregister():
	bpy.utils.unregister_class(OBJECT_OT_colorize_points)
