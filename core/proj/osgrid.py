
#Ordnance Survey National Grid references (e.g. TQ2580, TQ28SE) of British tiled datasets, such as
#the Environment Agency LiDAR tiles named TQ2580_P_8301_20120402_20120416.laz, which store no CRS.
#more infos : https://www.ordnancesurvey.co.uk/documents/resources/guide-to-nationalgrid.pdf

import os
import re

EPSG = 27700 #OSGB36 / British National Grid

#25 letters, I is not used
LETTERS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'

#two letters, 0 to 10 digits (even), optional quadrant, then the end of the name or a separator
TILE_NAME = re.compile(r'^([A-HJ-Z]{2})(\d{10}|\d{8}|\d{6}|\d{4}|\d{2})?(NE|NW|SE|SW)?(?=[^A-Z0-9]|$)', re.IGNORECASE)


def gridRefExtent(letters, digits='', quadrant=''):
	'''
	Extent (xmin, ymin, size) in meters of a grid reference: TQ is a 100 km square, TQ28 a 10 km
	square, TQ2580 a 1 km square, a quadrant (TQ28SE) is the quarter of the square. None if invalid
	'''
	letters = letters.upper()
	if len(letters) != 2 or any(letter not in LETTERS for letter in letters) or len(digits) % 2:
		return None
	l1, l2 = LETTERS.index(letters[0]), LETTERS.index(letters[1])
	#first letter: 500 km square from the false origin (S is 0, 0), second letter: 100 km square inside it
	e100k = ((l1 - 2) % 5) * 5 + l2 % 5
	n100k = (19 - (l1 // 5) * 5) - l2 // 5
	if not (0 <= e100k < 7 and 0 <= n100k < 13):
		return None
	half = len(digits) // 2
	size = 100000 / 10 ** half
	x = e100k * 100000 + (int(digits[:half]) * size if half else 0)
	y = n100k * 100000 + (int(digits[half:]) * size if half else 0)
	if quadrant:
		quadrant = quadrant.upper()
		size /= 2
		x += size if quadrant[1] == 'E' else 0
		y += size if quadrant[0] == 'N' else 0
	return x, y, size


def gridRefFromFileName(name):
	'''Grid reference starting a file name and its extent (ref, (xmin, ymin, size)), or None'''
	match = TILE_NAME.match(os.path.basename(name))
	if match is None:
		return None
	letters, digits, quadrant = match.group(1), match.group(2) or '', match.group(3) or ''
	extent = gridRefExtent(letters, digits, quadrant)
	if extent is None:
		return None
	return (letters + digits + quadrant).upper(), extent


def matchTile(name, mins, maxs, margin=0.1):
	'''
	Grid reference of a tile whose file name starts with it and whose xy bounds lie in its square
	(with a margin, as a fraction of the square size, for tiles overlapping their neighbours), or None.
	The bounds check rules out names that only look like a grid reference
	'''
	found = gridRefFromFileName(name)
	if found is None:
		return None
	ref, (x, y, size) = found
	tolerance = size * margin
	inside = (x - tolerance <= mins[0] and maxs[0] <= x + size + tolerance
		and y - tolerance <= mins[1] and maxs[1] <= y + size + tolerance)
	return ref if inside else None
