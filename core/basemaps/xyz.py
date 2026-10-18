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

'''XYZ tile url templates (as used by QGIS) as basemap sources of the MapService'''

import math
import hashlib

from .servicesDefs import SOURCES
from .mapservice import MapService, BBoxRequest

#web mercator ground resolution of zoom level 0, in meters per pixel for 256 px tiles
WM_RES0 = 2 * math.pi * 6378137 / 256


def registerXYZSource(url, zmin=0, zmax=19, referer='', name='XYZ'):
	'''
	Register an XYZ url template as a web mercator basemap source and return its key.
	Placeholders: {x} {y} {z}, {-y} for TMS rows (counted from the south), {q} for Bing like quadkeys
	'''
	key = 'XYZ_' + hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
	template = url.replace('{x}', '{X}').replace('{z}', '{Z}')
	grid, quadTree = 'WM', False
	if '{-y}' in template:
		template, grid = template.replace('{-y}', '{Y}'), 'WM_SW'
	else:
		template = template.replace('{y}', '{Y}')
	if '{q}' in template:
		template, quadTree = template.replace('{q}', '{QUADKEY}'), True
	SOURCES[key] = {
		"name": name,
		"description": url,
		"service": 'TMS',
		"grid": grid,
		"quadTree": quadTree,
		"layers": {
			"XYZ": {"urlKey": '', "name": name, "description": '', "format": 'png' if '.png' in url.lower() else 'jpeg', "zmin": zmin, "zmax": zmax}
		},
		"urlTemplate": template,
		"referer": referer,
	}
	return key

def isValidTemplate(url):
	return url.startswith(('http://', 'https://')) and (('{x}' in url and '{z}' in url and ('{y}' in url or '{-y}' in url)) or '{q}' in url)

def zoomForResolution(res, zmin=0, zmax=19):
	'''Lowest zoom level whose ground resolution (m/px) is at least as fine as res'''
	zoom = math.ceil(math.log2(WM_RES0 / res))
	return max(zmin, min(zmax, zoom))

def tileCount(bbox, zoom):
	'''Number of 256 px tiles covering a web mercator bbox (xmin, ymin, xmax, ymax) at a zoom level'''
	size = 256 * WM_RES0 / 2 ** zoom
	xmin, ymin, xmax, ymax = bbox
	return (math.floor(xmax / size) - math.floor(xmin / size) + 1) * (math.floor(ymax / size) - math.floor(ymin / size) + 1)

def fetchXYZ(srckey, bbox, zoom, cacheFolder, retries=2):
	"""
	Mosaic of the tiles covering a web mercator bbox, as a georeferenced NpImage.
	Tiles that failed to download are retried, the number still missing (shown with the
	mosaic background color) is returned with the image: (NpImage, nbMissingTiles)
	"""
	srv = MapService(srckey, cacheFolder)
	tiles = BBoxRequest(srv.srcTms, bbox, zoom).tiles
	srv.start()
	try:
		cache = srv.getCache('XYZ', False)
		for attempt in range(1 + retries):
			missing = cache.listMissingTiles(tiles)
			if not missing:
				break
			srv.seedTiles('XYZ', list(missing), toDstGrid=False, cpt=False)
		missing = cache.listMissingTiles(tiles)
		img = srv.getImage('XYZ', bbox, zoom, toDstGrid=False, cpt=False)
	finally:
		srv.stop()
	if img is None:
		raise IOError("Unable to download the basemap tiles")
	return img, len(missing)
