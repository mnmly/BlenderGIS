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

'''Import the layers of a QGIS project (.qgs / .qgz), each with its matching BlenderGIS importer'''

import os
import json
import logging
log = logging.getLogger(__name__)

import bpy
import pyproj #bundled as wheel (see blender_manifest.toml)
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, IntProperty, FloatProperty, EnumProperty, CollectionProperty
from bpy.types import Operator, PropertyGroup, UIList

from ..geoscene import GeoScene
from ..core.qgis import QgisProject, QgisProjectError
from ..core.qgis import project as qgis
from .utils import adjust3Dview, getBBOX

PKG = __package__.rsplit('.', 1)[0]

KIND_LABELS = {
	qgis.SHP: ('Shapefile', 'MESH_DATA'),
	qgis.GEOJSON: ('GeoJSON', 'MESH_DATA'),
	qgis.RASTER: ('Raster', 'IMAGE_DATA'),
	qgis.ASC: ('ASCII grid', 'MESH_GRID'),
	qgis.POINTCLOUD: ('Point cloud', 'POINTCLOUD_DATA'),
	qgis.XYZ: ('Web basemap', 'WORLD'),
	qgis.GPKG: ('GeoPackage', 'CANCEL'),
	qgis.UNSUPPORTED: ('Unsupported', 'CANCEL'),
}

#last parsed project, shared by the file dialog and the layers dialog
_project = None

def loadProject(filepath):
	global _project
	if _project is None or _project.path != filepath:
		_project = QgisProject.load(filepath)
	return _project


class BGIS_PG_qgis_layer(PropertyGroup):
	order: IntProperty() #index of the layer in QgisProject.layers
	label: StringProperty()
	kindLabel: StringProperty()
	icon: StringProperty()
	groupPath: StringProperty()
	selected: BoolProperty(name="Import", description="Import this layer")
	supported: BoolProperty()
	reason: StringProperty()


def fillLayerList(wm, project):
	'''List the project layers in the window manager collection, selecting the visible supported ones'''
	wm.bgis_qgis_layers.clear()
	wm.bgis_qgis_path = project.path
	for layer in project.layers:
		item = wm.bgis_qgis_layers.add()
		item.order = layer.order
		item.label = layer.name or layer.id or '?'
		item.kindLabel, item.icon = KIND_LABELS.get(layer.kind, KIND_LABELS[qgis.UNSUPPORTED])
		item.groupPath = ' / '.join(layer.groupPath)
		item.supported = layer.supported
		item.reason = layer.reason or ''
		item.selected = layer.supported and layer.visible


class BGIS_UL_qgis_layers(UIList):

	def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
		row = layout.row(align=True)
		check = row.row()
		check.enabled = item.supported
		check.prop(item, 'selected', text='')
		name = row.row()
		name.enabled = item.supported
		name.label(text=item.label, icon=item.icon)
		info = row.row()
		info.alignment = 'RIGHT'
		info.enabled = False
		info.label(text=item.reason if not item.supported else (item.groupPath or item.kindLabel))


class IMPORTGIS_OT_qgis_select(Operator):
	bl_idname = "importgis.qgis_select"
	bl_label = "Select layers"
	bl_description = "Select the layers to import"
	bl_options = {'INTERNAL'}

	mode: EnumProperty(items=[('VISIBLE', 'Visible', "Layers visible in QGIS"), ('ALL', 'All', ""), ('NONE', 'None', "")])

	def execute(self, context):
		wm = context.window_manager
		for item in wm.bgis_qgis_layers:
			if self.mode == 'VISIBLE':
				item.selected = item.supported and _project is not None and _project.layers[item.order].visible
			else:
				item.selected = item.supported and self.mode == 'ALL'
		return {'FINISHED'}


class IMPORTGIS_OT_qgis_file_dialog(Operator, ImportHelper):
	"""Import the layers of a QGIS project"""
	bl_idname = "importgis.qgis_file_dialog"
	bl_description = 'Import layers of a QGIS project (.qgs, .qgz)'
	bl_label = "Import QGIS project"
	bl_options = {'INTERNAL'}

	filter_glob: StringProperty(default="*.qgs;*.qgz", options={'HIDDEN'})

	@classmethod
	def poll(cls, context):
		return context.mode == 'OBJECT'

	def execute(self, context):
		try:
			project = loadProject(self.filepath)
		except QgisProjectError as e:
			self.report({'ERROR'}, str(e))
			return {'CANCELLED'}
		fillLayerList(context.window_manager, project)
		bpy.ops.importgis.qgis_project('INVOKE_DEFAULT', filepath=self.filepath)
		return {'FINISHED'}


class IMPORTGIS_OT_qgis_project(Operator):
	"""Import the selected layers of a QGIS project"""
	bl_idname = "importgis.qgis_project"
	bl_description = 'Import the selected layers of a QGIS project'
	bl_label = "Import QGIS project"
	bl_options = {'UNDO'}

	filepath: StringProperty(subtype='FILE_PATH', options={'HIDDEN'})

	point_size: FloatProperty(
		name="Point size",
		description="Size of the points of imported point clouds",
		default=0.5, min=0.01, max=10.0
		)

	@classmethod
	def poll(cls, context):
		return context.mode == 'OBJECT'

	def invoke(self, context, event):
		return context.window_manager.invoke_props_dialog(self, width=560)

	def draw(self, context):
		layout = self.layout
		wm = context.window_manager
		project = _project
		if project is not None:
			layout.label(text=f"{project.name}  ·  {project.crs or 'no CRS'}  {project.crsName or ''}", icon='FILE')
		layout.template_list('BGIS_UL_qgis_layers', '', wm, 'bgis_qgis_layers', wm, 'bgis_qgis_layers_index', rows=8)
		row = layout.row(align=True)
		row.label(text="Select:")
		for mode, text in (('VISIBLE', 'Visible in QGIS'), ('ALL', 'All'), ('NONE', 'None')):
			row.operator('importgis.qgis_select', text=text).mode = mode
		layout.prop(self, 'point_size')
		geoscn = GeoScene(context.scene)
		if geoscn.isGeoref:
			layout.label(text=f"Layers will be reprojected to the scene CRS {geoscn.crs}", icon='INFO')
		elif project is not None and project.crs:
			layout.label(text=f"Scene will use the project CRS {project.crs}", icon='INFO')

	def execute(self, context):
		wm = context.window_manager
		try:
			project = loadProject(self.filepath)
		except QgisProjectError as e:
			self.report({'ERROR'}, str(e))
			return {'CANCELLED'}
		if wm.bgis_qgis_path != project.path:
			fillLayerList(wm, project) #called without the dialog: visible layers
		layers = [project.layers[item.order] for item in wm.bgis_qgis_layers if item.selected and item.supported]
		if not layers:
			self.report({'WARNING'}, "No layer selected")
			return {'CANCELLED'}

		prefs = context.preferences.addons[PKG].preferences
		geoscn = GeoScene(context.scene)
		if geoscn.isBroken:
			self.report({'ERROR'}, "Scene georef is broken, please fix it beforehand")
			return {'CANCELLED'}
		try:
			setupScene(geoscn, project)
		except Exception as e:
			log.error("Cannot georeference the scene from the QGIS project", exc_info=True)
			self.report({'ERROR'}, f"Cannot set the scene CRS {project.crs}: {e}")
			return {'CANCELLED'}

		collections = {}
		imported, failed = [], []
		adjustView = prefs.adjust3Dview
		prefs.adjust3Dview = False #adjust once at the end
		try:
			#bottom layer first, like QGIS draws them
			for layer in reversed(layers):
				before = set(bpy.data.objects)
				try:
					result = importLayer(layer, prefs, self.point_size)
				except Exception as e:
					log.error(f"Cannot import QGIS layer {layer.name}", exc_info=True)
					failed.append(f"{layer.name} ({e})")
					continue
				objs = [obj for obj in bpy.data.objects if obj not in before]
				if result != {'FINISHED'} or not objs:
					failed.append(f"{layer.name} (importer returned {result})")
					continue
				organizeObjects(context, objs, layer, project, collections)
				imported.extend(objs)
		finally:
			prefs.adjust3Dview = adjustView

		if imported:
			bpy.ops.object.select_all(action='DESELECT')
			for obj in imported:
				obj.select_set(True)
			context.view_layer.objects.active = imported[-1]
			if adjustView and context.screen is not None:
				bbox = getBBOX.fromObj(imported[0])
				for obj in imported[1:]:
					bbox += getBBOX.fromObj(obj)
				adjust3Dview(context, bbox)

		skipped = [l for l in project.layers if not l.supported]
		msg = f"Imported {len(layers) - len(failed)} of {len(layers)} selected layers"
		if failed:
			msg += "; failed: " + ", ".join(failed)
			self.report({'WARNING'}, msg)
		else:
			self.report({'INFO'}, msg)
		for layer in skipped:
			log.info(f"QGIS layer {layer.name} skipped: {layer.reason}")
		return {'FINISHED'} if imported else {'CANCELLED'}


def setupScene(geoscn, project):
	'''Georeference an empty scene like the QGIS project: project CRS, origin at the center of the map view'''
	if geoscn.isGeoref or not project.crs:
		return
	if not geoscn.hasCRS:
		geoscn.crs = project.crs
	center = project.canvasCenter
	if center is None:
		return #the first imported layer will set the origin
	if geoscn.crs != project.crs:
		center = pyproj.Transformer.from_crs(project.crs, geoscn.crs, always_xy=True).transform(*center)
	geoscn.setOriginPrj(*center)


def ensurePredefCRS(prefs, crs, name):
	'''Add crs to the user predefined CRS list (needed by the importers CRS dropdowns)'''
	data = json.loads(prefs.predefCrsJson)
	if any(item[0] == crs for item in data):
		return
	data.append([crs, name or crs, "Added from a QGIS project"])
	prefs.predefCrsJson = json.dumps(data)


def importLayer(layer, prefs, pointSize):
	'''Run the BlenderGIS importer matching the layer kind'''
	if layer.kind == qgis.SHP:
		return bpy.ops.importgis.shapefile(filepath=layer.path, shpCRS=layer.crs)
	if layer.kind == qgis.GEOJSON:
		return bpy.ops.importgis.geojson(filepath=layer.path, shpCRS=layer.crs)
	ensurePredefCRS(prefs, layer.crs, layer.crsName)
	if layer.kind == qgis.RASTER:
		return bpy.ops.importgis.georaster(filepath=layer.path, importMode='PLANE', reprojection=True, rastCRS=layer.crs)
	if layer.kind == qgis.ASC:
		return bpy.ops.importgis.asc_file(filepath=layer.path, fileCRS=layer.crs)
	if layer.kind == qgis.POINTCLOUD:
		#the CRS QGIS uses for the layer, it also covers files without CRS in their header
		return bpy.ops.importgis.laz(filepath=layer.path, files=[{'name': os.path.basename(layer.path)}],
			crsSource='CUSTOM', customCRS=layer.crs, point_size=pointSize)
	raise ValueError(f"Unsupported layer kind {layer.kind}")


def organizeObjects(context, objs, layer, project, collections):
	'''Name, tag, color and move the objects of an imported layer into collections mirroring the QGIS groups'''
	collection = getGroupCollection(context, project, layer.groupPath, collections)
	for obj in objs:
		if len(objs) == 1:
			obj.name = layer.name
		obj['qgis_layer'] = layer.name
		obj['qgis_project'] = project.path
		for coll in list(obj.users_collection):
			coll.objects.unlink(obj)
		collection.objects.link(obj)
		if layer.color is not None and obj.type == 'MESH' and layer.kind in (qgis.SHP, qgis.GEOJSON):
			setLayerColor(obj, layer)

def getGroupCollection(context, project, groupPath, collections):
	'''Collection for a QGIS group path, created under a collection named after the project'''
	if () not in collections:
		root = bpy.data.collections.new(project.name)
		context.scene.collection.children.link(root)
		collections[()] = root
	for depth in range(1, len(groupPath) + 1):
		path = groupPath[:depth]
		if path not in collections:
			coll = bpy.data.collections.new(path[-1])
			collections[path[:-1]].children.link(coll)
			collections[path] = coll
	return collections[tuple(groupPath)]

def setLayerColor(obj, layer):
	mat = bpy.data.materials.new('QGIS ' + layer.name)
	mat.diffuse_color = layer.color
	mat.use_nodes = True
	principled = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
	if principled is not None:
		principled.inputs['Base Color'].default_value = layer.color
	obj.data.materials.append(mat)
	obj.color = layer.color


classes = [BGIS_PG_qgis_layer, BGIS_UL_qgis_layers, IMPORTGIS_OT_qgis_select, IMPORTGIS_OT_qgis_file_dialog, IMPORTGIS_OT_qgis_project]

def register():
	for cls in classes:
		try:
			bpy.utils.register_class(cls)
		except ValueError:
			log.warning('{} is already registered, now unregister and retry... '.format(cls))
			bpy.utils.unregister_class(cls)
			bpy.utils.register_class(cls)
	bpy.types.WindowManager.bgis_qgis_layers = CollectionProperty(type=BGIS_PG_qgis_layer)
	bpy.types.WindowManager.bgis_qgis_layers_index = IntProperty()
	bpy.types.WindowManager.bgis_qgis_path = StringProperty()

def unregister():
	del bpy.types.WindowManager.bgis_qgis_layers
	del bpy.types.WindowManager.bgis_qgis_layers_index
	del bpy.types.WindowManager.bgis_qgis_path
	for cls in reversed(classes):
		bpy.utils.unregister_class(cls)
