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
import math
import functools
import threading
import logging
log = logging.getLogger(__name__)

import bpy
import bpy.utils.previews
import numpy as np
import pyproj #bundled as wheel (see blender_manifest.toml)
import pyproj.aoi
import pyproj.database
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, IntProperty, FloatProperty, FloatVectorProperty, EnumProperty, CollectionProperty
from bpy.types import Operator, PropertyGroup, UIList

from ..geoscene import GeoScene
from ..core.qgis import QgisProject, QgisProjectError
from ..core.qgis import project as qgis
from ..core.basemaps.xyz import registerXYZSource, zoomForResolution, tileCount, fetchXYZ, XYZPrefetch, WM_RES0
from .utils import adjust3Dview, getBBOX, placeObj, addTexture
from .utils.point_markers import setPointMarkersMaterial

PKG = __package__.rsplit('.', 1)[0]

#raster planes are stacked in QGIS drawing order, this far apart (scene units), to avoid z-fighting
RASTER_STACK_GAP = 0.05
#refuse basemaps needing more tiles than this (rate limits of tile servers)
MAX_TILES = 2000
#above this many tiles, the region dialog tells how long the basemap download may take
LARGE_DOWNLOAD = 300
TILES_PER_SECOND = 20 #measured with 10 download threads, tiles not cached yet

def downloadEstimate(nbTiles):
	seconds = nbTiles / TILES_PER_SECOND
	return f"~{round(seconds / 60)} min" if seconds >= 90 else f"~{round(seconds, -1):.0f} s"

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

REGION_MODES = [
	('CANVAS', 'QGIS map view', "Extent of the map view saved in the QGIS project"),
	('LAYERS', 'Selected layers', "Extent of the selected layers, basemaps excluded"),
	('CUSTOM', 'Custom', "Bounds typed in the project CRS"),
]

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
	if project.canvasExtent is not None:
		wm.bgis_qgis_region = 'CANVAS'
		applyRegionPreset(wm, project)


def selectedLayers(wm, project):
	return [project.layers[item.order] for item in wm.bgis_qgis_layers if item.selected and item.supported]

def layersExtent(layers, crs):
	'''Union of the layers extents, in crs (None if no layer has an extent)'''
	boxes = []
	for layer in layers:
		if layer.kind == qgis.XYZ or layer.extent is None or not layer.crs:
			continue
		try:
			boxes.append(pyproj.Transformer.from_crs(layer.crs, crs, always_xy=True).transform_bounds(*layer.extent, densify_pts=21))
		except pyproj.exceptions.ProjError:
			log.warning(f"Cannot reproject the extent of QGIS layer {layer.name}")
	if not boxes:
		return None
	return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))

def presetExtent(wm, project, mode):
	if mode == 'CANVAS':
		return project.canvasExtent
	if mode == 'LAYERS':
		return layersExtent(selectedLayers(wm, project), project.crs)
	return None

#set while a preset writes the region bounds, so that only edits by the user switch to custom
_syncingRegion = False

def applyRegionPreset(wm, project):
	'''Copy the extent of the selected preset to the region bounds'''
	global _syncingRegion
	extent = presetExtent(wm, project, wm.bgis_qgis_region) if project is not None else None
	if extent is None:
		return
	_syncingRegion = True
	try:
		wm.bgis_qgis_region_bbox = extent
	finally:
		_syncingRegion = False

def updateRegionMode(wm, context):
	applyRegionPreset(wm, _project)

def updateRegionBbox(wm, context):
	if not _syncingRegion and wm.bgis_qgis_region != 'CUSTOM':
		wm.bgis_qgis_region = 'CUSTOM'

def basemapPlan(extent, crs, xyz, basemapSize):
	'''Web mercator bbox, zoom level and number of tiles of an XYZ layer download covering extent (in crs)'''
	bbox = pyproj.Transformer.from_crs(crs, 'EPSG:3857', always_xy=True).transform_bounds(*extent, densify_pts=21)
	zoom = zoomForResolution(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / basemapSize, xyz['zmin'], xyz['zmax'])
	return bbox, zoom, tileCount(bbox, zoom)

def groundSize(extent, crs):
	'''Width and height in meters of extent (in crs), measured through its center'''
	xmin, ymin, xmax, ymax = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform_bounds(*extent, densify_pts=21)
	geod = pyproj.Geod(ellps='WGS84')
	cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
	return geod.inv(xmin, cy, xmax, cy)[2], geod.inv(cx, ymin, cx, ymax)[2]

def formatDistance(meters):
	return f"{meters / 1000:.2f} km" if meters >= 1000 else f"{meters:.0f} m"


#Region preview: a small basemap mosaic around the region, downloaded in a thread,
#with the download area and the region drawn over it at each dialog redraw
PREVIEW_SIZE = 512 #pixels, largest side of the preview mosaic
PREVIEW_MARGIN = 0.25 #context around the map view / layers / region, relative to their size
PREVIEW_SCALE = 20 #size of the preview in the dialog, in UI units
_preview = {'pcoll': None, 'base': None, 'baseBbox': None, 'baseKey': None, 'fetching': None, 'error': None, 'missing': 0, 'drawnKey': None}

def tileAligned(bbox, zoom):
	'''Web mercator bbox snapped outward to the tiles of a zoom level, the area actually downloaded'''
	size = 256 * WM_RES0 / 2 ** zoom
	xmin, ymin, xmax, ymax = bbox
	return (math.floor(xmin / size) * size, math.floor(ymin / size) * size,
		(math.floor(xmax / size) + 1) * size, (math.floor(ymax / size) + 1) * size)

def regionOutline(extent, crs, nbPts=16):
	'''Region rectangle (in crs) as a closed web mercator polyline, edges densified (they can curve once reprojected)'''
	xmin, ymin, xmax, ymax = extent
	ts = [i / nbPts for i in range(nbPts)]
	pts = [(xmin + (xmax - xmin) * t, ymin) for t in ts] + [(xmax, ymin + (ymax - ymin) * t) for t in ts] \
		+ [(xmax - (xmax - xmin) * t, ymax) for t in ts] + [(xmin, ymax - (ymax - ymin) * t) for t in ts]
	pts.append(pts[0])
	xs, ys = pyproj.Transformer.from_crs(crs, 'EPSG:3857', always_xy=True).transform([p[0] for p in pts], [p[1] for p in pts])
	return list(zip(xs, ys))

def previewContext(wm, project, extent):
	'''Web mercator bbox shown by the preview: map view, selected layers and region, with a margin'''
	boxes = [b for b in (project.canvasExtent, layersExtent(selectedLayers(wm, project), project.crs), extent) if b is not None]
	toWM = pyproj.Transformer.from_crs(project.crs, 'EPSG:3857', always_xy=True)
	boxes = [toWM.transform_bounds(*b, densify_pts=21) for b in boxes]
	xmin, ymin = min(b[0] for b in boxes), min(b[1] for b in boxes)
	xmax, ymax = max(b[2] for b in boxes), max(b[3] for b in boxes)
	m = max(xmax - xmin, ymax - ymin) * PREVIEW_MARGIN
	return (xmin - m, ymin - m, xmax + m, ymax + m)

def _fetchPreview(key, srckey, bbox, zoom, cacheFolder):
	'''Thread: download the preview mosaic, as float RGBA rows bottom up (the layout of preview pixels)'''
	try:
		mosaic, nbMissing = fetchXYZ(srckey, bbox, zoom, cacheFolder, retries=0)
		data = mosaic.data
		if data.ndim == 2:
			data = data[:, :, None]
		if data.shape[2] < 3:
			data = np.repeat(data[:, :, :1], 3, axis=2)
		rgba = np.ones(data.shape[:2] + (4,), dtype=np.float32)
		rgba[:, :, :3] = data[:, :, :3] / 255
		georefBbox = mosaic.georef.bbox
		result = rgba[::-1].copy(), (georefBbox.xmin, georefBbox.ymin, georefBbox.xmax, georefBbox.ymax), None, nbMissing
	except Exception as e:
		log.warning("Cannot download the basemap region preview", exc_info=True)
		result = None, None, str(e), 0
	if _preview['fetching'] == key: #else a newer request is running
		_preview['base'], _preview['baseBbox'], _preview['error'], _preview['missing'] = result
		_preview['baseKey'] = key if result[0] is not None else None
		_preview['drawnKey'] = None
		_preview['fetching'] = None

def requestPreview(wm, project, extent, layer, cacheFolder):
	'''Start downloading a preview mosaic if the current one does not cover the region'''
	ctx = previewContext(wm, project, extent)
	base = _preview['baseBbox']
	covered = base is not None and _preview['baseKey'] is not None and _preview['baseKey'][0] == layer.xyz['url'] \
		and base[0] <= ctx[0] and base[1] <= ctx[1] and base[2] >= ctx[2] and base[3] >= ctx[3]
	if covered or _preview['fetching'] is not None:
		return
	zoom = zoomForResolution(max(ctx[2] - ctx[0], ctx[3] - ctx[1]) / PREVIEW_SIZE, layer.xyz['zmin'], layer.xyz['zmax'])
	key = (layer.xyz['url'], ctx, zoom)
	if _preview['error'] is not None and _preview.get('lastKey') == key:
		return #do not retry the same failed request at each redraw
	srckey = registerXYZSource(layer.xyz['url'], layer.xyz['zmin'], layer.xyz['zmax'], layer.xyz['referer'], layer.name)
	_preview['fetching'] = key
	_preview['lastKey'] = key
	_preview['error'] = None
	threading.Thread(target=_fetchPreview, args=(key, srckey, ctx, zoom, cacheFolder), daemon=True).start()

def _drawPolyline(img, pts, color, width=2):
	'''Draw a polyline of pixel coordinates (x right, y up) on an RGBA float image'''
	h, w = img.shape[:2]
	for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
		n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
		xs = np.linspace(x0, x1, n).round().astype(int)
		ys = np.linspace(y0, y1, n).round().astype(int)
		for dx in range(-(width // 2), width - width // 2):
			for dy in range(-(width // 2), width - width // 2):
				ok = (xs + dx >= 0) & (xs + dx < w) & (ys + dy >= 0) & (ys + dy < h)
				img[ys[ok] + dy, xs[ok] + dx, :3] = color

def previewIcon(extent, crs, downloadBbox):
	'''Icon of the preview mosaic with the download area and the region drawn over it (None until downloaded)'''
	base, bbox = _preview['base'], _preview['baseBbox']
	if base is None:
		return None
	if _preview['pcoll'] is None:
		_preview['pcoll'] = bpy.utils.previews.new()
	pcoll = _preview['pcoll']
	preview = pcoll.get('region') or pcoll.new('region')
	key = (_preview['baseKey'], extent, downloadBbox)
	if _preview['drawnKey'] != key:
		h, w = base.shape[:2]
		toPx = lambda x, y: ((x - bbox[0]) / (bbox[2] - bbox[0]) * w, (y - bbox[1]) / (bbox[3] - bbox[1]) * h)
		img = base.copy()
		#dim what will not be downloaded
		x0, y0 = toPx(downloadBbox[0], downloadBbox[1])
		x1, y1 = toPx(downloadBbox[2], downloadBbox[3])
		inside = np.zeros((h, w), dtype=bool)
		inside[max(0, int(y0)):max(0, int(math.ceil(y1))), max(0, int(x0)):max(0, int(math.ceil(x1)))] = True
		img[~inside, :3] *= 0.35
		_drawPolyline(img, [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], (1, 1, 1), width=1)
		_drawPolyline(img, [toPx(x, y) for x, y in regionOutline(extent, crs)], (1, 0.55, 0.1), width=2)
		preview.image_size = (w, h)
		preview.image_pixels_float.foreach_set(img.ravel())
		_preview['drawnKey'] = key
	return preview.icon_id

def freePreview():
	if _preview['pcoll'] is not None:
		bpy.utils.previews.remove(_preview['pcoll'])
	_preview.update(pcoll=None, base=None, baseBbox=None, baseKey=None, fetching=None, error=None, missing=0, drawnKey=None)


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
		description="Maximum size in pixels of the basemaps (XYZ tiles) downloaded for the region",
		default=2048, min=256, max=8192
		)

	basemap_extent: FloatVectorProperty(
		name="Basemap region",
		description="Bounds (xmin, ymin, xmax, ymax) in the project CRS of the downloaded basemaps, all zero for the QGIS map view",
		size=4, options={'HIDDEN'}
		)

	#set when run from the layers dialog: selected basemaps then open the region dialog before importing
	ask_region: BoolProperty(options={'HIDDEN', 'SKIP_SAVE'})

	#set when the basemap tiles were just downloaded in the background: no retry of the missing ones
	prefetched: BoolProperty(options={'HIDDEN', 'SKIP_SAVE'})

	@classmethod
	def poll(cls, context):
		return context.mode == 'OBJECT'

	def invoke(self, context, event):
		self.ask_region = True
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
		geoscn = GeoScene(context.scene)
		if geoscn.isGeoref:
			layout.label(text=f"Layers will be reprojected to the scene CRS {geoscn.crs}", icon='INFO')
		elif project is not None and project.crs:
			sceneCrs = sceneCrsFor(project.crs, project.canvasCenter)
			if sceneCrs != project.crs:
				layout.label(text=f"Scene will use {sceneCrs} (UTM zone of the map view), the project CRS {project.crs} is in degrees", icon='INFO')
			else:
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
		layers = selectedLayers(wm, project)
		if not bpy.app.online_access:
			layers = [layer for layer in layers if layer.kind != qgis.XYZ]
		if not layers:
			self.report({'WARNING'}, "No layer selected")
			return {'CANCELLED'}
		if self.ask_region and any(layer.kind == qgis.XYZ for layer in layers):
			#nothing imported yet, the region dialog runs this operator again
			bpy.ops.importgis.qgis_basemap_region('INVOKE_DEFAULT', filepath=self.filepath,
				point_size=self.point_size, basemap_size=self.basemap_size)
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


class IMPORTGIS_OT_qgis_basemap_region(Operator):
	"""Choose the region of the basemaps downloaded with a QGIS project, then import it"""
	bl_idname = "importgis.qgis_basemap_region"
	bl_description = 'Choose the region of the basemaps to download, then import the QGIS project'
	bl_label = "Basemap region"
	bl_options = {'INTERNAL', 'UNDO'}

	filepath: StringProperty(subtype='FILE_PATH', options={'HIDDEN'})
	point_size: FloatProperty(options={'HIDDEN'})
	basemap_size: IntProperty(
		name="Basemap size",
		description="Maximum size in pixels of the basemaps (XYZ tiles) downloaded for the region",
		default=2048, min=256, max=8192
		)

	#redraw the dialog when a value changes (live preview of the download)
	def check(self, context):
		return True

	def invoke(self, context, event):
		wm = context.window_manager
		if _project is None or not _project.crs:
			self.report({'ERROR'}, "The project has no CRS to define the basemap region in")
			return {'CANCELLED'}
		applyRegionPreset(wm, _project) #the layer selection may have changed the layers extent
		return wm.invoke_props_dialog(self, width=460, confirm_text="Import")

	def region(self, context):
		'''Region bounds, or None with the reason they cannot be used'''
		wm = context.window_manager
		if wm.bgis_qgis_region != 'CUSTOM' and presetExtent(wm, _project, wm.bgis_qgis_region) is None:
			return None, "No extent found for this preset, type the bounds"
		xmin, ymin, xmax, ymax = wm.bgis_qgis_region_bbox
		if xmin >= xmax or ymin >= ymax:
			return None, "X min / Y min must be lower than X max / Y max"
		return (xmin, ymin, xmax, ymax), None

	def draw(self, context):
		layout = self.layout
		wm = context.window_manager
		layout.row().prop(wm, 'bgis_qgis_region', expand=True)
		self.drawPreview(context, layout)
		col = layout.column(align=True)
		row = col.row(align=True)
		row.prop(wm, 'bgis_qgis_region_bbox', index=0, text="X min")
		row.prop(wm, 'bgis_qgis_region_bbox', index=2, text="X max")
		row = col.row(align=True)
		row.prop(wm, 'bgis_qgis_region_bbox', index=1, text="Y min")
		row.prop(wm, 'bgis_qgis_region_bbox', index=3, text="Y max")
		layout.label(text=f"Bounds in the project CRS {_project.crs}")
		layout.prop(self, 'basemap_size')

		box = layout.box()
		extent, error = self.region(context)
		if error:
			box.label(text=error, icon='ERROR')
			return
		try:
			w, h = groundSize(extent, _project.crs)
			box.label(text=f"Region {formatDistance(w)} × {formatDistance(h)}", icon='SELECT_SET')
			for layer in selectedLayers(wm, _project):
				if layer.kind != qgis.XYZ:
					continue
				bbox, zoom, nbTiles = basemapPlan(extent, _project.crs, layer.xyz, self.basemap_size)
				tooMany = nbTiles > MAX_TILES
				text = f"{layer.name}: zoom {zoom} · {nbTiles} tiles"
				box.label(text=text + (f" (more than {MAX_TILES})" if tooMany else ""), icon='ERROR' if tooMany else 'WORLD')
				if not tooMany and nbTiles > LARGE_DOWNLOAD:
					box.label(text=f"Downloads in the background, up to {downloadEstimate(nbTiles)} (Esc cancels)", icon='TIME')
		except pyproj.exceptions.ProjError as e:
			box.label(text=f"Cannot reproject the region: {e}", icon='ERROR')

	def drawPreview(self, context, layout):
		box = layout.box()
		extent, error = self.region(context)
		xyzLayers = [layer for layer in selectedLayers(context.window_manager, _project) if layer.kind == qgis.XYZ]
		if error or not xyzLayers:
			box.label(text="No preview", icon='IMAGE_DATA')
			return
		layer = xyzLayers[0]
		try:
			prefs = context.preferences.addons[PKG].preferences
			requestPreview(context.window_manager, _project, extent, layer, prefs.cacheFolder)
			bbox, zoom, nbTiles = basemapPlan(extent, _project.crs, layer.xyz, self.basemap_size)
			icon = previewIcon(extent, _project.crs, tileAligned(bbox, zoom))
		except pyproj.exceptions.ProjError as e:
			box.label(text=f"Cannot reproject the region: {e}", icon='ERROR')
			return
		if icon is None:
			if _preview['error']:
				box.label(text=f"Preview unavailable: {_preview['error']}", icon='ERROR')
			else:
				box.label(text=f"Loading {layer.name} preview... (move the mouse to refresh)", icon='TIME')
			return
		box.template_icon(icon_value=icon, scale=PREVIEW_SCALE)
		if _preview['missing']:
			box.label(text=f"{_preview['missing']} preview tiles could not be downloaded (shown grey)", icon='ERROR')
		row = box.row()
		row.label(text="Region", icon='SELECT_SET')
		row.label(text="Downloaded tiles", icon='MESH_PLANE')

	def execute(self, context):
		extent, error = self.region(context)
		if error:
			self.report({'ERROR'}, error)
			return {'CANCELLED'}
		args = dict(filepath=self.filepath, point_size=self.point_size, basemap_size=self.basemap_size, basemap_extent=extent)
		if bpy.app.background or context.window is None:
			return bpy.ops.importgis.qgis_project('EXEC_DEFAULT', **args)
		#with a UI, download the basemaps in the background first: Blender stays usable
		return bpy.ops.importgis.qgis_download('INVOKE_DEFAULT', **args)


#background basemap download of the running import (one at a time)
_download = None

class IMPORTGIS_OT_qgis_download(Operator):
	"""Download the basemap tiles of a QGIS project in the background, then import it"""
	bl_idname = "importgis.qgis_download"
	bl_label = "Download basemaps"
	bl_options = {'INTERNAL', 'UNDO'}

	filepath: StringProperty(subtype='FILE_PATH', options={'HIDDEN'})
	point_size: FloatProperty(options={'HIDDEN'})
	basemap_size: IntProperty(options={'HIDDEN'})
	basemap_extent: FloatVectorProperty(size=4, options={'HIDDEN'})

	def invoke(self, context, event):
		global _download
		if _download is not None and _download.is_alive():
			self.report({'ERROR'}, "A basemap download is already running")
			return {'CANCELLED'}
		wm = context.window_manager
		project = loadProject(self.filepath)
		jobs = []
		for layer in selectedLayers(wm, project):
			if layer.kind != qgis.XYZ:
				continue
			bbox, zoom, nbTiles = basemapPlan(tuple(self.basemap_extent), project.crs, layer.xyz, self.basemap_size)
			if nbTiles > MAX_TILES:
				continue #refused by the import, which reports it
			srckey = registerXYZSource(layer.xyz['url'], layer.xyz['zmin'], layer.xyz['zmax'], layer.xyz['referer'], layer.name)
			jobs.append((layer.name, srckey, bbox, zoom))
		prefs = context.preferences.addons[PKG].preferences
		_download = XYZPrefetch(jobs, prefs.cacheFolder)
		_download.start()
		self._timer = wm.event_timer_add(0.2, window=context.window)
		wm.modal_handler_add(self)
		return {'RUNNING_MODAL'}

	def modal(self, context, event):
		if event.type == 'ESC' and event.value == 'PRESS':
			_download.cancel()
			self.finish(context)
			self.report({'WARNING'}, "QGIS project import cancelled")
			return {'CANCELLED'}
		if event.type != 'TIMER':
			return {'PASS_THROUGH'} #keep Blender usable during the download
		if _download.is_alive():
			progress = _download.progress()
			if progress is not None:
				name, done, total = progress
				context.workspace.status_text_set(f"BlenderGIS: downloading {name}, {done} / {total} tiles (Esc to cancel)")
			return {'PASS_THROUGH'}
		self.finish(context)
		return bpy.ops.importgis.qgis_project('EXEC_DEFAULT', filepath=self.filepath, point_size=self.point_size,
			basemap_size=self.basemap_size, basemap_extent=self.basemap_extent, prefetched=True)

	def finish(self, context):
		context.window_manager.event_timer_remove(self._timer)
		context.workspace.status_text_set(None)


@functools.lru_cache(maxsize=8)
def sceneCrsFor(crs, center):
	'''
	CRS of an empty scene importing a project in crs: the project CRS, or when it is in degrees
	(1 scene unit would be 1 degree) the UTM zone of center, the map view center in crs
	'''
	try:
		projectCrs = pyproj.CRS.from_user_input(crs)
	except pyproj.exceptions.CRSError:
		return crs
	if not projectCrs.is_geographic or center is None:
		return crs
	lon, lat = pyproj.Transformer.from_crs(projectCrs, 'EPSG:4326', always_xy=True).transform(*center)
	utm = pyproj.database.query_utm_crs_info(datum_name='WGS 84', area_of_interest=pyproj.aoi.AreaOfInterest(lon, lat, lon, lat))
	return f"{utm[0].auth_name}:{utm[0].code}" if utm else crs

def setupScene(geoscn, project):
	'''Georeference an empty scene like the QGIS project: project CRS (UTM if in degrees), origin at the center of the map view'''
	if geoscn.isGeoref or not project.crs:
		return
	if not geoscn.hasCRS:
		geoscn.crs = sceneCrsFor(project.crs, project.canvasCenter)
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
	Download the XYZ tiles covering the basemap region (the QGIS map view by default) and drape them on a grid mesh.
	The tiles are web mercator: the grid vertices are reprojected to the scene CRS with pyproj
	(no image warping needed), its uv map stays linear in web mercator space.
	'''
	if not bpy.app.online_access:
		raise IOError("Online access is disabled")
	extent = tuple(options.basemap_extent) if any(options.basemap_extent) else project.canvasExtent
	if extent is None or not project.crs:
		raise ValueError("The project has no map view extent to download the basemap for")
	geoscn = GeoScene(context.scene)
	xyz = layer.xyz
	bbox, zoom, nbTiles = basemapPlan(extent, project.crs, xyz, options.basemap_size)
	if nbTiles > MAX_TILES:
		raise ValueError(f"{nbTiles} tiles needed at zoom {zoom}, reduce the basemap size or the region")

	srckey = registerXYZSource(xyz['url'], xyz['zmin'], xyz['zmax'], xyz['referer'], layer.name)
	mosaic, nbMissing = fetchXYZ(srckey, bbox, zoom, prefs.cacheFolder, retries=0 if options.prefetched else 2)
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
	setPointMarkersMaterial(obj, mat)


classes = [BGIS_PG_qgis_layer, BGIS_UL_qgis_layers, IMPORTGIS_OT_qgis_select, IMPORTGIS_OT_qgis_file_dialog, IMPORTGIS_OT_qgis_project,
	IMPORTGIS_OT_qgis_basemap_region, IMPORTGIS_OT_qgis_download]

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
	bpy.types.WindowManager.bgis_qgis_region = EnumProperty(name="Region", description="Region of the downloaded basemaps",
		items=REGION_MODES, update=updateRegionMode)
	bpy.types.WindowManager.bgis_qgis_region_bbox = FloatVectorProperty(name="Region bounds",
		description="Bounds (xmin, ymin, xmax, ymax) of the basemap region in the project CRS", size=4, precision=6, update=updateRegionBbox)

def unregister():
	if _download is not None and _download.is_alive():
		_download.cancel()
	freePreview()
	del bpy.types.WindowManager.bgis_qgis_layers
	del bpy.types.WindowManager.bgis_qgis_layers_index
	del bpy.types.WindowManager.bgis_qgis_path
	del bpy.types.WindowManager.bgis_qgis_region
	del bpy.types.WindowManager.bgis_qgis_region_bbox
	for cls in reversed(classes):
		bpy.utils.unregister_class(cls)
