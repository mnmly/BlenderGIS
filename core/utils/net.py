import ssl

import logging
log = logging.getLogger(__name__)

def _createSSLContext():
	'''
	HTTPS context used for every request made by the addon.
	Python's default certificate store is often empty in Blender's bundled interpreter (macOS, Windows),
	so prefer the certifi bundle shipped with Blender instead of disabling verification process-wide.
	'''
	try:
		import certifi
	except ImportError:
		log.debug('certifi unavailable, using the default SSL context')
		return ssl.create_default_context()
	return ssl.create_default_context(cafile=certifi.where())

SSL_CONTEXT = _createSSLContext()
