import bpy

def getAppData():
	'''Per-user writable folder (logs, default basemaps cache) managed by Blender's extension system'''
	return bpy.utils.extension_path_user(__package__, create=True)
