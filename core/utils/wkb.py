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

'''
Decode WKB geometries (ISO and EWKB flavors, as returned by GDAL/OGR) into GeoJSON like dicts.
Standard library only. M values are dropped, Z values are kept.
'''

import struct

GEOJSON_TYPES = {1: 'Point', 2: 'LineString', 3: 'Polygon', 4: 'MultiPoint', 5: 'MultiLineString', 6: 'MultiPolygon', 7: 'GeometryCollection'}

#EWKB (PostGIS) flags
EWKB_Z = 0x80000000
EWKB_M = 0x40000000
EWKB_SRID = 0x20000000


class WKBError(ValueError):
	pass


class _Reader():

	def __init__(self, data):
		self.data = bytes(data)
		self.pos = 0

	def unpack(self, fmt):
		size = struct.calcsize(fmt)
		if self.pos + size > len(self.data):
			raise WKBError("Truncated WKB")
		values = struct.unpack_from(fmt, self.data, self.pos)
		self.pos += size
		return values

	def geometry(self):
		order, = self.unpack('B')
		endian = '<' if order == 1 else '>'
		wkbType, = self.unpack(endian + 'I')
		hasZ = bool(wkbType & EWKB_Z)
		hasM = bool(wkbType & EWKB_M)
		if wkbType & EWKB_SRID:
			self.unpack(endian + 'I')
		wkbType &= 0x0FFFFFFF
		#ISO WKB: 1000 Z, 2000 M, 3000 ZM
		dims, base = divmod(wkbType, 1000)
		hasZ = hasZ or dims in (1, 3)
		hasM = hasM or dims in (2, 3)
		if base not in GEOJSON_TYPES:
			raise WKBError(f"Unsupported WKB geometry type {wkbType} (curves are not supported)")
		nbCoords = 2 + hasZ + hasM
		pointFmt = endian + 'd' * nbCoords
		keep = 3 if hasZ else 2 #drop M

		def point():
			return list(self.unpack(pointFmt)[:keep])
		def points():
			n, = self.unpack(endian + 'I')
			return [point() for _ in range(n)]
		def count():
			return self.unpack(endian + 'I')[0]

		geomType = GEOJSON_TYPES[base]
		if base == 1:
			coords = point()
			if all(c != c for c in coords): #NaN coordinates: empty point
				return {'type': geomType, 'coordinates': []}
			return {'type': geomType, 'coordinates': coords}
		if base == 2:
			return {'type': geomType, 'coordinates': points()}
		if base == 3:
			return {'type': geomType, 'coordinates': [points() for _ in range(count())]}
		parts = [self.geometry() for _ in range(count())]
		if base == 7:
			return {'type': geomType, 'geometries': parts}
		return {'type': geomType, 'coordinates': [part['coordinates'] for part in parts]}


def wkbToGeojson(data):
	'''GeoJSON like dict {'type': ..., 'coordinates': ...} of a WKB geometry, None for an empty input'''
	if data is None or len(data) == 0:
		return None
	return _Reader(data).geometry()
