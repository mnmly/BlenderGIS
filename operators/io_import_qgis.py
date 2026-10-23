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
import re
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
from ..core.basemaps.xyz import registerXYZSource, zoomForResolution, tileCount, fetchXYZ
from .utils import adjust3Dview, getBBOX, placeObj, addTexture

PKG = __package__.rsplit('.', 1)[0]

#raster planes are stacked in QGIS drawing order, this far apart (scene units), to avoid z-fighting
RASTER_STACK_GAP = 0.05
#refuse basemaps needing more tiles than this (rate limits of tile servers)
MAX_TILES = 2000

KIND_LABELS = {
	qgis.SHP: ('Shapefile', 'MESH_DATA'),
	qgis.GEOJSON: ('GeoJSON', 'MESH_DATA'),
	qgis.RASTER: ('Raster', 'IMAGE_DATA'),
	qgis.ASC: ('ASCII grid', 'MESH_GRID'),
	qgis.POINTCLOUD: ('Point cloud', 'POINTCLOUD_DATA'),
	qgis.XYZ: ('Web basemap', 'WORLD'),
	qgis.OGR: ('Vector (GDAL)', 'MESH_DATA'),
	qgis.UNSUPPORTED: ('Unsupported', 'CANCEL'),
}

#last parsed project, shared by the file dialog and the layers dialog
_project = None

def resolveCrs(wkt, proj4):
	'''AUTH:CODE (or proj4) of a custom QGIS CRS, keeping the horizontal part of a compound CRS'''
	for definition in (wkt, proj4):
		if not definition:
			continue
		try:
			crs = pyproj.CRS.from_user_input(definition)
		except pyproj.exceptions.CRSError:
			continue
		if crs.is_compound:
			crs = crs.sub_crs_list[0]
		authority = crs.to_authority(min_confidence=70)
		return ':'.join(authority) if authority else crs.to_proj4()
	return None

def loadProject(filepath, reload=False):
	'''Parsed project, kept between the file dialog and the layers dialog (whose list indexes its layers)'''
	global _project
	if reload or _project is None or _project.path != filepath:
		_project = QgisProject.load(filepath, resolveCrs)
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
		if layer.kind == qgis.XYZ and item.supported and not bpy.app.online_access:
			item.supported = False
			item.reason = "Enable Allow Online Access (Preferences > System) to download basemaps"
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
			project = loadProject(self.filepath, reload=True) #the file may have been edited in QGIS since the last import
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

	basemap_size: IntProperty(
		name="Basemap size",
		description="Maximum size in pixels of the basemaps (XYZ tiles) downloaded for the QGIS map view extent",
		default=2048, min=256, max=8192
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
		row = layout.row()
		row.prop(self, 'point_size')
		row.prop(self, 'basemap_size')
		geoscn = GeoScene(context.scene)
		if geoscn.isGeoref:
			layout.label(text=f"Layers will be reprojected to the scene CRS {geoscn.crs}", icon='INFO')
		elif project is not None and project.crs:
			layout.label(text=f"Scene will use the project CRS {project.crs}", icon='INFO')

	def execute(self, context):
		wm = context.window_manager
		try:
			project = loadProject(self.filepath, reload=wm.bgis_qgis_path != self.filepath)
		except QgisProjectError as e:
			self.report({'ERROR'}, str(e))
			return {'CANCELLED'}
		if wm.bgis_qgis_path != project.path:
			fillLayerList(wm, project) #called without the dialog: visible layers
		layers = [project.layers[item.order] for item in wm.bgis_qgis_layers if item.selected and item.supported]
		if not bpy.app.online_access:
			layers = [layer for layer in layers if layer.kind != qgis.XYZ]
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
		imported, failed, importedLayers = [], [], []
		adjustView = prefs.adjust3Dview
		prefs.adjust3Dview = False #adjust once at the end
		try:
			#bottom layer first, like QGIS draws them
			for layer in reversed(layers):
				before = set(bpy.data.objects)
				try:
					result = importLayer(context, layer, project, prefs, self)
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
				importedLayers.append((layer, objs))
		finally:
			prefs.adjust3Dview = adjustView

		#raster planes all lie at z=0: offset them in QGIS drawing order (top layer highest)
		planes = [(layer, objs) for layer, objs in importedLayers if layer.kind in (qgis.RASTER, qgis.XYZ)]
		for rank, (layer, objs) in enumerate(sorted(planes, key=lambda lo: lo[0].order)):
			for obj in objs:
				obj.location.z -= rank * RASTER_STACK_GAP

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


def importLayer(context, layer, project, prefs, options):
	'''Run the BlenderGIS importer matching the layer kind'''
	if layer.kind == qgis.XYZ:
		return importXYZ(context, layer, project, prefs, options)
	if layer.kind == qgis.SHP:
		return bpy.ops.importgis.shapefile(filepath=layer.path, shpCRS=layer.crs)
	if layer.kind == qgis.GEOJSON:
		return bpy.ops.importgis.geojson(filepath=layer.path, shpCRS=layer.crs)
	if layer.kind == qgis.OGR:
		ogr = layer.ogr
		return bpy.ops.importgis.ogr(filepath=layer.path, shpCRS=layer.crs, layer='' if ogr['layer'] is None else str(ogr['layer']),
			where=ogr['where'] or '', geometryType=ogr['geometryType'] or '')
	ensurePredefCRS(prefs, layer.crs, layer.crsName)
	if layer.kind == qgis.RASTER:
		return bpy.ops.importgis.georaster(filepath=layer.path, importMode='PLANE', reprojection=True, rastCRS=layer.crs)
	if layer.kind == qgis.ASC:
		return bpy.ops.importgis.asc_file(filepath=layer.path, fileCRS=layer.crs)
	if layer.kind == qgis.POINTCLOUD:
		#the CRS QGIS uses for the layer, it also covers files without CRS in their header
		return bpy.ops.importgis.laz(filepath=layer.path, files=[{'name': os.path.basename(layer.path)}],
			crsSource='CUSTOM', customCRS=layer.crs, point_size=options.point_size)
	raise ValueError(f"Unsupported layer kind {layer.kind}")


def importXYZ(context, layer, project, prefs, options, gridSize=32):
	'''
	Download the XYZ tiles covering the QGIS map view and drape them on a grid mesh.
	The tiles are web mercator: the grid vertices are reprojected to the scene CRS with pyproj
	(no image warping needed), its uv map stays linear in web mercator space.
	'''
	if not bpy.app.online_access:
		raise IOError("Online access is disabled")
	if project.canvasExtent is None or not project.crs:
		raise ValueError("The project has no map view extent to download the basemap for")
	geoscn = GeoScene(context.scene)
	xyz = layer.xyz
	toWM = pyproj.Transformer.from_crs(project.crs, 'EPSG:3857', always_xy=True)
	bbox = toWM.transform_bounds(*project.canvasExtent, densify_pts=21)
	zoom = zoomForResolution(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / options.basemap_size, xyz['zmin'], xyz['zmax'])
	if tileCount(bbox, zoom) > MAX_TILES:
		raise ValueError(f"{tileCount(bbox, zoom)} tiles needed at zoom {zoom}, reduce the basemap size or the QGIS map view")

	srckey = registerXYZSource(xyz['url'], xyz['zmin'], xyz['zmax'], xyz['referer'], layer.name)
	mosaic, nbMissing = fetchXYZ(srckey, bbox, zoom, prefs.cacheFolder)
	if nbMissing:
		options.report({'WARNING'}, f"{layer.name}: {nbMissing} tiles could not be downloaded (shown grey), import again to retry")

	#keep the mosaic with its world file next to the tiles cache, the plane image and Colorize read it
	folder = os.path.join(prefs.cacheFolder, 'qgis_basemaps')
	os.makedirs(folder, exist_ok=True)
	path = os.path.join(folder, f"{re.sub(r'[^A-Za-z0-9_-]+', '_', layer.name)}_z{zoom}.png")
	mosaic.save(path)
	img = bpy.data.images.load(path, check_existing=True)
	img.reload()

	#grid over the mosaic extent, from web mercator to scene coordinates
	extent = mosaic.georef.bbox #tile aligned, a bit larger than the map view
	xmin, ymin, xmax, ymax = extent.xmin, extent.ymin, extent.xmax, extent.ymax
	steps = [i / gridSize for i in range(gridSize + 1)]
	wmPts = [(xmin + (xmax - xmin) * u, ymin + (ymax - ymin) * v) for v in steps for u in steps]
	toScene = pyproj.Transformer.from_crs('EPSG:3857', geoscn.crs, always_xy=True)
	xs, ys = toScene.transform([p[0] for p in wmPts], [p[1] for p in wmPts])
	dx, dy = geoscn.getOriginPrj()
	verts = [(x - dx, y - dy, 0) for x, y in zip(xs, ys)]
	n = gridSize + 1
	faces = [(j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i) for j in range(gridSize) for i in range(gridSize)]
	mesh = bpy.data.meshes.new(layer.name)
	mesh.from_pydata(verts, [], faces)
	uvLayer = mesh.uv_layers.new(name='rastUVmap')
	uvs = [(i / gridSize, j / gridSize) for j in range(n) for i in range(n)]
	for loop in mesh.loops:
		uvLayer.data[loop.index].uv = uvs[loop.vertex_index]
	mesh.update()

	obj = placeObj(mesh, layer.name)
	mat = bpy.data.materials.new('rastMat')
	obj.data.materials.append(mat)
	addTexture(mat, img, uvLayer, name='rastText')
	obj['bgis_raster_path'] = path
	obj['bgis_raster_crs'] = 'EPSG:3857'
	obj['bgis_basemap_zoom'] = zoom
	return {'FINISHED'}


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
		if layer.color is not None and obj.type == 'MESH' and layer.kind in (qgis.SHP, qgis.GEOJSON, qgis.OGR):
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
