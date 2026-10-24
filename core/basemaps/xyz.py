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
import threading
import logging
log = logging.getLogger(__name__)

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

def downloadTiles(srv, tiles, retries=2, cpt=False):
	'''Download the tiles missing from the cache of a started XYZ MapService, retrying the failed ones. Return the tiles still missing'''
	cache = srv.getCache('XYZ', False)
	for attempt in range(1 + retries):
		missing = cache.listMissingTiles(tiles)
		if not missing or not srv.running:
			break
		srv.seedTiles('XYZ', list(missing), toDstGrid=False, cpt=cpt)
	return cache.listMissingTiles(tiles)

class XYZPrefetch(threading.Thread):
	'''
	Download in a thread, into the tiles cache, the tiles of XYZ basemaps: jobs = [(name, srckey, bbox, zoom)].
	fetchXYZ then builds the mosaics from the cache without waiting for the network.
	'''
	def __init__(self, jobs, cacheFolder, retries=2):
		super().__init__(daemon=True)
		self.jobs, self.cacheFolder, self.retries = jobs, cacheFolder, retries
		self.current = None #(name, MapService) of the running download
		self.cancelled = False
		self.error = None

	def run(self):
		try:
			for name, srckey, bbox, zoom in self.jobs:
				if self.cancelled:
					break
				srv = MapService(srckey, self.cacheFolder)
				tiles = BBoxRequest(srv.srcTms, bbox, zoom).tiles
				self.current = (name, srv)
				srv.start()
				try:
					downloadTiles(srv, tiles, self.retries, cpt=True)
				finally:
					srv.stop()
		except Exception as e:
			log.error("Cannot download the basemap tiles", exc_info=True)
			self.error = str(e)
		self.current = None

	def cancel(self):
		self.cancelled = True
		current = self.current
		if current is not None:
			current[1].stop() #download threads check it between tiles

	def progress(self):
		'''(basemap name, tiles done, tiles to download) of the running download, or None'''
		current = self.current
		if current is None:
			return None
		name, srv = current
		return name, srv.cptTiles, srv.nbTiles

def fetchXYZ(srckey, bbox, zoom, cacheFolder, retries=2):
	"""
	Mosaic of the tiles covering a web mercator bbox, as a georeferenced NpImage.
	Tiles that failed to download are retried, the number still missing (shown with the
	mosaic background color) is returned with the image: (NpImage, nbMissingTiles).
	Raise IOError when no tile at all could be downloaded (wrong url, server down, offline).
	"""
	srv = MapService(srckey, cacheFolder)
	tiles = BBoxRequest(srv.srcTms, bbox, zoom).tiles
	srv.start()
	try:
		missing = downloadTiles(srv, tiles, retries)
		if tiles and len(missing) == len(tiles):
			host = SOURCES[srckey]['urlTemplate'].split('/')[2]
			raise IOError(f"No tile could be downloaded from {host}, check the url and the connection")
		img = srv.getImage('XYZ', bbox, zoom, toDstGrid=False, cpt=False)
	finally:
		srv.stop()
	if img is None:
		raise IOError("Unable to download the basemap tiles")
	return img, len(missing)
