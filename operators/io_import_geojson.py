# -*- coding:utf-8 -*-
import os, sys, json, time
import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty
from bpy.types import Operator
import bmesh
import math
from mathutils import Vector

import logging
log = logging.getLogger(__name__)

from ..geoscene import GeoScene, georefManagerLayout
from ..prefs import PredefCRS
from ..core import BBOX
from ..core.proj import Reproj
from ..core.utils import perf_clock

from .utils import adjust3Dview, getBBOX, DropToGround

PKG = __package__.rsplit('.', 1)[0]


#GeoJSON geometry types mapped to the three primitives the bmesh builder understands
#(same categories as the shapefile importer : Point / PolyLine / Polygon)
GEOM_CATEGORY = {
	'Point': 'Point',
	'MultiPoint': 'Point',
	'LineString': 'PolyLine',
	'MultiLineString': 'PolyLine',
	'Polygon': 'Polygon',
	'MultiPolygon': 'Polygon',
}


def loadGeojson(filepath):
	"""Read a GeoJSON file and return the parsed python object"""
	with open(filepath, 'r', encoding='utf-8') as f:
		return json.load(f)


def getCRSFromGeojson(data):
	"""Extract an EPSG code from a legacy GeoJSON 'crs' member.
	RFC 7946 mandates WGS84 (EPSG:4326) and drops the crs member, so this is
	only a best effort for older files. Returns an 'EPSG:xxxx' string or None."""
	crs = data.get('crs')
	if not crs:
		return None
	try:
		name = crs['properties']['name']
	except (KeyError, TypeError):
		return None
	#typical forms : "urn:ogc:def:crs:EPSG::4326", "EPSG:4326", "urn:ogc:def:crs:OGC:1.3:CRS84"
	name = name.upper()
	if 'CRS84' in name:
		return 'EPSG:4326'
	if 'EPSG' in name:
		code = name.replace('::', ':').rstrip(':').split(':')[-1]
		if code.isdigit():
			return 'EPSG:' + code
	return None


def iterFeatures(data):
	"""Normalize any GeoJSON root (FeatureCollection / Feature / bare Geometry /
	GeometryCollection) into an iterator of (geometry, properties) tuples."""
	t = data.get('type')
	if t == 'FeatureCollection':
		for feat in data.get('features', []):
			yield from iterFeatures(feat)
	elif t == 'Feature':
		geom = data.get('geometry')
		props = data.get('properties') or {}
		if geom is None:
			return
		if geom.get('type') == 'GeometryCollection':
			for g in geom.get('geometries', []):
				yield (g, props)
		else:
			yield (geom, props)
	elif t == 'GeometryCollection':
		for g in data.get('geometries', []):
			yield (g, {})
	elif t in GEOM_CATEGORY:
		yield (data, {})


def geomToParts(geom):
	"""Convert a GeoJSON geometry to (category, parts) where category is one of
	Point/PolyLine/Polygon and parts is a list of coordinate rings/lines, each
	being a list of [x, y] or [x, y, z] positions."""
	t = geom.get('type')
	coords = geom.get('coordinates')
	category = GEOM_CATEGORY.get(t)
	if category is None or coords is None:
		return None, []

	if t == 'Point':
		return 'Point', [[coords]]
	if t == 'MultiPoint':
		return 'Point', [coords]
	if t == 'LineString':
		return 'PolyLine', [coords]
	if t == 'MultiLineString':
		return 'PolyLine', coords
	if t == 'Polygon':
		return 'Polygon', coords #list of rings (exterior first, then holes)
	if t == 'MultiPolygon':
		rings = []
		for poly in coords:
			rings.extend(poly)
		return 'Polygon', rings
	return None, []


def listFeatureFields(filepath):
	"""Collect the union of all feature property keys (preserving first-seen order)
	along with whether each one only holds numeric values."""
	try:
		data = loadGeojson(filepath)
	except Exception:
		log.warning("Unable to read GeoJSON fields", exc_info=True)
		return {}
	fields = {} #name -> isNumeric
	for geom, props in iterFeatures(data):
		for k, v in props.items():
			numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
			if k not in fields:
				fields[k] = numeric
			elif not numeric and v is not None:
				fields[k] = False
	return fields



class IMPORTGIS_OT_geojson_file_dialog(Operator):
	"""Select a GeoJSON file then start the importgis.geojson_props_dialog operator"""

	bl_idname = "importgis.geojson_file_dialog"
	bl_description = 'Import GeoJSON (.geojson .json)'
	bl_label = "Import GeoJSON"
	bl_options = {'INTERNAL'}

	# Import dialog properties
	filepath: StringProperty(
		name="File Path",
		description="Filepath used for importing the file",
		maxlen=1024,
		subtype='FILE_PATH' )

	filename_ext = ".geojson"

	filter_glob: StringProperty(
			default = "*.geojson;*.json",
			options = {'HIDDEN'} )

	def invoke(self, context, event):
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def draw(self, context):
		layout = self.layout

	def execute(self, context):
		if os.path.exists(self.filepath):
			bpy.ops.importgis.geojson_props_dialog('INVOKE_DEFAULT', filepath=self.filepath)
		else:
			self.report({'ERROR'}, "Invalid filepath")
		return{'FINISHED'}



class IMPORTGIS_OT_geojson_props_dialog(Operator):
	"""GeoJSON importer properties dialog"""

	bl_idname = "importgis.geojson_props_dialog"
	bl_description = 'Import GeoJSON (.geojson .json)'
	bl_label = "Import GeoJSON"
	bl_options = {"INTERNAL"}

	filepath: StringProperty()

	#special function to auto redraw an operator popup called through invoke_props_dialog
	def check(self, context):
		return True

	def listFields(self, context):
		fieldsItems = []
		for name in listFeatureFields(self.filepath):
			#put each item in a tuple (key, label, tooltip)
			fieldsItems.append( (name, name, '') )
		return fieldsItems

	# GeoJSON CRS definition
	def listPredefCRS(self, context):
		return PredefCRS.getEnumItems()

	def listObjects(self, context):
		objs = []
		for index, object in enumerate(bpy.context.scene.objects):
			if object.type == 'MESH':
				#put each object in a tuple (key, label, tooltip) and add this to the objects list
				objs.append((object.name, object.name, "Object named " + object.name))
		return objs

	reprojection: BoolProperty(
			name="Specifiy GeoJSON CRS",
			description="Specifiy GeoJSON CRS if it's different from scene CRS",
			default=False )

	shpCRS: EnumProperty(
		name = "GeoJSON CRS",
		description = "Choose a Coordinate Reference System",
		items = listPredefCRS)

	# Elevation source
	vertsElevSource: EnumProperty(
			name="Elevation source",
			description="Select the source of vertices z value",
			items=[
			('NONE', 'None', "Flat geometry"),
			('GEOM', 'Geometry', "Use z value from geometry coordinates if exists"),
			('FIELD', 'Field', "Extract z elevation value from a property field"),
			('OBJ', 'Object', "Get z elevation value from an existing ground mesh")
			],
			default='GEOM')

	# Elevation object
	objElevLst: EnumProperty(
		name="Elev. object",
		description="Choose the mesh from which extract z elevation",
		items=listObjects )

	# Elevation field
	fieldElevName: EnumProperty(
		name = "Elev. field",
		description = "Choose field",
		items = listFields )

	#Extrusion field
	useFieldExtrude: BoolProperty(
			name="Extrusion from field",
			description="Extract z extrusion value from a property field",
			default=False )

	fieldExtrudeName: EnumProperty(
		name = "Field",
		description = "Choose field",
		items = listFields )

	#Extrusion axis
	extrusionAxis: EnumProperty(
			name="Extrude along",
			description="Select extrusion axis",
			items=[ ('Z', 'z axis', "Extrude along Z axis"),
			('NORMAL', 'Normal', "Extrude along normal")] )

	#Create separate objects
	separateObjects: BoolProperty(
			name="Separate objects",
			description="Warning : can be very slow with lot of features",
			default=False )

	#Name objects from field
	useFieldName: BoolProperty(
			name="Object name from field",
			description="Extract name for created objects from a property field",
			default=False )
	fieldObjName: EnumProperty(
		name = "Field",
		description = "Choose field",
		items = listFields )


	def draw(self, context):
		#Function used by blender to draw the panel.
		scn = context.scene
		layout = self.layout

		layout.prop(self, 'vertsElevSource')
		if self.vertsElevSource == 'FIELD':
			layout.prop(self, 'fieldElevName')
		elif self.vertsElevSource == 'OBJ':
			layout.prop(self, 'objElevLst')
		#
		layout.prop(self, 'useFieldExtrude')
		if self.useFieldExtrude:
			layout.prop(self, 'fieldExtrudeName')
			layout.prop(self, 'extrusionAxis')
		#
		layout.prop(self, 'separateObjects')
		if self.separateObjects:
			layout.prop(self, 'useFieldName')
		else:
			self.useFieldName = False
		if self.separateObjects and self.useFieldName:
			layout.prop(self, 'fieldObjName')
		#
		geoscn = GeoScene()
		if geoscn.isPartiallyGeoref:
			layout.prop(self, 'reprojection')
			if self.reprojection:
				self.crsInputLayout(context)
			#
			georefManagerLayout(self, context)
		else:
			self.crsInputLayout(context)


	def crsInputLayout(self, context):
		layout = self.layout
		row = layout.row(align=True)
		split = row.split(factor=0.35, align=True)
		split.label(text='CRS:')
		split.prop(self, "shpCRS", text='')
		row.operator("bgis.add_predef_crs", text='', icon='ADD')


	def invoke(self, context, event):
		#Default the CRS picker to the file's declared CRS (or WGS84 per RFC 7946)
		#Only works if that CRS is part of the predefined list, otherwise keep the default
		try:
			crs = getCRSFromGeojson(loadGeojson(self.filepath)) or 'EPSG:4326'
			if PredefCRS.getName(crs) is not None:
				self.shpCRS = crs
		except Exception:
			log.warning("Unable to preset GeoJSON CRS", exc_info=True)
		return context.window_manager.invoke_props_dialog(self)

	def execute(self, context):

		elevField = self.fieldElevName if self.vertsElevSource == 'FIELD' else ""
		extrudField = self.fieldExtrudeName if self.useFieldExtrude else ""
		nameField = self.fieldObjName if self.useFieldName else ""
		if self.vertsElevSource == 'OBJ':
			if not self.objElevLst:
				self.report({'ERROR'}, "No elevation object")
				return {'CANCELLED'}
			else:
				objElevName = self.objElevLst
		else:
			objElevName = '' #will not be used

		geoscn = GeoScene()
		if geoscn.isBroken:
			self.report({'ERROR'}, "Scene georef is broken, please fix it beforehand")
			return {'CANCELLED'}

		if geoscn.isGeoref:
			if self.reprojection:
				shpCRS = self.shpCRS
			else:
				shpCRS = geoscn.crs
		else:
			shpCRS = self.shpCRS

		try:
			bpy.ops.importgis.geojson('INVOKE_DEFAULT', filepath=self.filepath, shpCRS=shpCRS, elevSource=self.vertsElevSource,
				fieldElevName=elevField, objElevName=objElevName, fieldExtrudeName=extrudField, fieldObjName=nameField,
				extrusionAxis=self.extrusionAxis, separateObjects=self.separateObjects)
		except Exception as e:
			log.error('GeoJSON import fails', exc_info=True)
			self.report({'ERROR'}, 'GeoJSON import fails, check logs.')
			return {'CANCELLED'}

		return{'FINISHED'}


class IMPORTGIS_OT_geojson(Operator):
	"""Import from GeoJSON file format (.geojson .json)"""

	bl_idname = "importgis.geojson"
	bl_description = 'Import GeoJSON (.geojson .json)'
	bl_label = "Import GeoJSON"
	bl_options = {"UNDO"}

	filepath: StringProperty()

	shpCRS: StringProperty(name = "GeoJSON CRS", description = "Coordinate Reference System")

	elevSource: StringProperty(name = "Elevation source", description = "Elevation source", default='GEOM') # [NONE, GEOM, OBJ, FIELD]
	objElevName: StringProperty(name = "Elevation object name", description = "")

	fieldElevName: StringProperty(name = "Elevation field", description = "Field name")
	fieldExtrudeName: StringProperty(name = "Extrusion field", description = "Field name")
	fieldObjName: StringProperty(name = "Objects names field", description = "Field name")

	#Extrusion axis
	extrusionAxis: EnumProperty(
			name="Extrude along",
			description="Select extrusion axis",
			items=[ ('Z', 'z axis', "Extrude along Z axis"),
			('NORMAL', 'Normal', "Extrude along normal")]
			)
	#Create separate objects
	separateObjects: BoolProperty(
			name="Separate objects",
			description="Import to separate objects instead one large object",
			default=False
			)

	@classmethod
	def poll(cls, context):
		return context.mode == 'OBJECT'

	def __del__(self):
		bpy.context.window.cursor_set('DEFAULT')

	def execute(self, context):

		prefs = bpy.context.preferences.addons[PKG].preferences

		#Set cursor representation to 'loading' icon
		w = context.window
		w.cursor_set('WAIT')
		t0 = perf_clock()

		bpy.ops.object.select_all(action='DESELECT')

		#Path
		layerName = os.path.splitext(os.path.basename(self.filepath))[0]

		#Read geojson
		log.info("Read GeoJSON...")
		try:
			data = loadGeojson(self.filepath)
		except Exception as e:
			log.error("Unable to read GeoJSON", exc_info=True)
			self.report({'ERROR'}, "Unable to read GeoJSON, check logs")
			return {'CANCELLED'}

		#Materialize features (geometry, properties) and figure out property fields
		features = list(iterFeatures(data))
		if not features:
			self.report({'ERROR'}, "No feature found in GeoJSON")
			return {'CANCELLED'}

		fieldsInfo = listFeatureFields(self.filepath) #name -> isNumeric

		if self.fieldObjName and self.separateObjects:
			if self.fieldObjName not in fieldsInfo:
				self.report({'ERROR'}, "Unable to find name field")
				return {'CANCELLED'}

		if self.fieldElevName:
			if self.fieldElevName not in fieldsInfo:
				self.report({'ERROR'}, "Unable to find elevation field")
				return {'CANCELLED'}
			if not fieldsInfo[self.fieldElevName]:
				self.report({'ERROR'}, "Elevation field do not contains numeric values")
				return {'CANCELLED'}

		if self.fieldExtrudeName:
			if self.fieldExtrudeName not in fieldsInfo:
				self.report({'ERROR'}, "Unable to find extrusion field")
				return {'CANCELLED'}
			if not fieldsInfo[self.fieldExtrudeName]:
				self.report({'ERROR'}, "Extrusion field do not contains numeric values")
				return {'CANCELLED'}

		if self.elevSource == 'OBJ':
			scn = bpy.context.scene
			elevObj = scn.objects[self.objElevName]
			rayCaster = DropToGround(scn, elevObj)

		#Get geojson and scene georef infos
		shpCRS = self.shpCRS
		geoscn = GeoScene()
		if geoscn.isBroken:
			self.report({'ERROR'}, "Scene georef is broken, please fix it beforehand")
			return {'CANCELLED'}

		if not geoscn.hasCRS:
			try:
				geoscn.crs = shpCRS
			except Exception as e:
				log.error("Cannot set scene crs", exc_info=True)
				self.report({'ERROR'}, "Cannot set scene crs, check logs for more infos")
				return {'CANCELLED'}

		#Init reprojector class
		if geoscn.crs != shpCRS:
			log.info("Data will be reprojected from {} to {}".format(shpCRS, geoscn.crs))
			try:
				rprj = Reproj(shpCRS, geoscn.crs)
			except Exception as e:
				log.error('Reprojection fails', exc_info=True)
				self.report({'ERROR'}, "Unable to reproject data, check logs for more infos.")
				return {'CANCELLED'}
			if rprj.iproj == 'EPSGIO':
				if len(features) > 100:
					self.report({'ERROR'}, "Reprojection through online epsg.io engine is limited to 100 features. \nPlease install GDAL or pyproj module.")
					return {'CANCELLED'}

		#Compute source bbox over all coordinates
		def iterCoords():
			for geom, props in features:
				_, parts = geomToParts(geom)
				for part in parts:
					for pt in part:
						yield pt
		xmin = ymin = math.inf
		xmax = ymax = -math.inf
		for pt in iterCoords():
			x, y = pt[0], pt[1]
			if x < xmin: xmin = x
			if x > xmax: xmax = x
			if y < ymin: ymin = y
			if y > ymax: ymax = y
		if xmin == math.inf:
			self.report({'ERROR'}, "No valid geometry found in GeoJSON")
			return {'CANCELLED'}

		bbox = BBOX(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
		if geoscn.crs != shpCRS:
			bbox = rprj.bbox(bbox)

		#Get or set georef dx, dy
		if not geoscn.isGeoref:
			dx, dy = bbox.center
			geoscn.setOriginPrj(dx, dy)
		else:
			dx, dy = geoscn.getOriginPrj()

		nbFeats = len(features)

		#Create an empty BMesh
		bm = bmesh.new()
		#Extrusion is exponentially slow with large bmesh
		#it's fastest to extrude a small bmesh and then join it to a final large bmesh
		if not self.separateObjects and self.fieldExtrudeName:
			finalBm = bmesh.new()

		progress = -1

		if self.separateObjects:
			layer = bpy.data.collections.new(layerName)
			context.scene.collection.children.link(layer)

		#Main iteration over features
		for i, (geom, record) in enumerate(features):

			geomCategory, parts = geomToParts(geom)
			if geomCategory is None:
				log.warning('Skipping unsupported geometry type for feature {} : {}'.format(i, geom.get('type')))
				continue

			#Progress infos
			pourcent = round(((i+1)*100)/nbFeats)
			if pourcent in list(range(0, 110, 10)) and pourcent != progress:
				progress = pourcent
				if pourcent == 100:
					print(str(pourcent)+'%')
				else:
					print(str(pourcent), end="%, ")
				sys.stdout.flush()

			#Get extrusion offset
			if self.fieldExtrudeName:
				try:
					offset = float(record[self.fieldExtrudeName])
				except Exception as e:
					log.warning('Cannot extract extrusion value for feature {} : {}'.format(i, e))
					offset = 0 #null values will be set to zero

			#Iter over parts
			for part in parts:

				nbPts = len(part)
				if nbPts == 0:
					continue

				#Reproj geom (x,y only, keep z)
				if geoscn.crs != shpCRS:
					pts2d = rprj.pts([(pt[0], pt[1]) for pt in part])
				else:
					pts2d = [(pt[0], pt[1]) for pt in part]

				# EXTRACT 3D GEOM
				geom3d = [] #will contains a list of 3d points
				for k, pt in enumerate(part):

					if self.elevSource == 'OBJ':
						rcHit = rayCaster.rayCast(x=pts2d[k][0]-dx, y=pts2d[k][1]-dy)
						z = rcHit.loc.z #will be automatically set to zero if not rcHit.hit

					elif self.elevSource == 'FIELD':
						try:
							z = float(record[self.fieldElevName])
						except Exception as e:
							log.warning('Cannot extract elevation value for feature {} : {}'.format(i, e))
							z = 0 #null values will be set to zero

					elif self.elevSource == 'GEOM' and len(pt) >= 3:
						z = pt[2]

					else:
						z = 0

					#Shift coords
					geom3d.append((pts2d[k][0]-dx, pts2d[k][1]-dy, z))


				# BUILD BMESH

				# POINTS
				if geomCategory == 'Point':
					vert = [bm.verts.new(pt) for pt in geom3d]
					#Extrusion
					if self.fieldExtrudeName and offset > 0:
						vect = (0, 0, offset) #along Z
						result = bmesh.ops.extrude_vert_indiv(bm, verts=vert)
						verts = result['verts']
						bmesh.ops.translate(bm, verts=verts, vec=vect)

				# LINES
				if geomCategory == 'PolyLine':
					verts = [bm.verts.new(pt) for pt in geom3d]
					edges = []
					for n in range(len(geom3d)-1):
						edge = bm.edges.new( [verts[n], verts[n+1] ])
						edges.append(edge)
					#Extrusion
					if self.fieldExtrudeName and offset > 0:
						vect = (0, 0, offset) # along Z
						result = bmesh.ops.extrude_edge_only(bm, edges=edges)
						verts = [elem for elem in result['geom'] if isinstance(elem, bmesh.types.BMVert)]
						bmesh.ops.translate(bm, verts=verts, vec=vect)

				# NGONS
				if geomCategory == 'Polygon':
					#GeoJSON (RFC 7946) exterior rings are counterclockwise, which is already face-up in Blender
					#Drop the last point because it duplicates the first one (closed ring)
					ring = list(geom3d)
					if len(ring) > 1 and ring[0] == ring[-1]:
						ring.pop()
					if len(ring) >= 3: #needs 3 points to get a valid face
						verts = [bm.verts.new(pt) for pt in ring]
						face = bm.faces.new(verts)
						#update normal to avoid null vector
						face.normal_update()
						if face.normal.z < 0: #this is a polygon hole, bmesh cannot handle polygon hole
							pass #TODO
						#Extrusion
						if self.fieldExtrudeName and offset > 0:
							#build translate vector
							if self.extrusionAxis == 'NORMAL':
								normal = face.normal
								vect = normal * offset
							elif self.extrusionAxis == 'Z':
								vect = (0, 0, offset)
							faces = bmesh.ops.extrude_discrete_faces(bm, faces=[face]) #return {'faces': [BMFace]}
							verts = faces['faces'][0].verts
							if self.elevSource == 'OBJ':
								# Making flat roof
								z = max([v.co.z for v in verts]) + offset #get max z coord
								for v in verts:
									v.co.z = z
							else:
								bmesh.ops.translate(bm, verts=verts, vec=vect)


			if self.separateObjects:

				if self.fieldObjName:
					try:
						name = record[self.fieldObjName]
					except Exception as e:
						log.warning('Cannot extract name value for feature {} : {}'.format(i, e))
						name = ''
					name = '' if name is None else str(name)
				else:
					name = layerName

				#Calc bmesh bbox
				_bbox = getBBOX.fromBmesh(bm)

				#Calc bmesh geometry origin and translate coords according to it
				#then object location will be set to initial bmesh origin
				ox, oy, oz = _bbox.center
				oz = _bbox.zmin
				bmesh.ops.translate(bm, verts=bm.verts, vec=(-ox, -oy, -oz))

				#Create new mesh from bmesh
				mesh = bpy.data.meshes.new(name)
				bm.to_mesh(mesh)
				bm.clear()

				#Validate new mesh
				mesh.validate(verbose=False)

				#Place obj
				obj = bpy.data.objects.new(name, mesh)
				layer.objects.link(obj)
				context.view_layer.objects.active = obj
				obj.select_set(True)
				obj.location = (ox, oy, oz)

				#write attributes data
				for fieldName, v in record.items():
					if v is None:
						continue
					if isinstance(v, bool):
						obj[fieldName] = v
					elif isinstance(v, (int, float)):
						#cast to float to avoid overflow error when affecting custom property
						obj[fieldName] = float(v)
					elif isinstance(v, str):
						obj[fieldName] = v
					else:
						obj[fieldName] = str(v)

			elif self.fieldExtrudeName:
				#Join to final bmesh (use from_mesh method hack)
				buff = bpy.data.meshes.new(".temp")
				bm.to_mesh(buff)
				finalBm.from_mesh(buff)
				bpy.data.meshes.remove(buff)
				bm.clear()

		#Write back the whole mesh
		if not self.separateObjects:

			mesh = bpy.data.meshes.new(layerName)

			if self.fieldExtrudeName:
				bm.free()
				bm = finalBm

			if prefs.mergeDoubles:
				bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
			bm.to_mesh(mesh)

			#Finish
			mesh.validate(verbose=False) #return true if the mesh has been corrected
			obj = bpy.data.objects.new(layerName, mesh)
			context.scene.collection.objects.link(obj)
			context.view_layer.objects.active = obj
			obj.select_set(True)
			bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')

		#free the bmesh
		bm.free()

		t = perf_clock() - t0
		log.info('Build in %f seconds' % t)

		#Adjust grid size
		if prefs.adjust3Dview:
			bbox.shift(-dx, -dy) #convert bbox in 3d view space
			adjust3Dview(context, bbox)


		return {'FINISHED'}


classes = [
	IMPORTGIS_OT_geojson_file_dialog,
	IMPORTGIS_OT_geojson_props_dialog,
	IMPORTGIS_OT_geojson
]

def register():
	for cls in classes:
		try:
			bpy.utils.register_class(cls)
		except ValueError as e:
			log.warning('{} is already registered, now unregister and retry... '.format(cls))
			bpy.utils.unregister_class(cls)
			bpy.utils.register_class(cls)

def unregister():
	for cls in classes:
		bpy.utils.unregister_class(cls)
