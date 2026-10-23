Blender GIS
==========
Supported Blender versions : 4.3 to 5.x, on Windows, macOS and Linux, installed as an extension.

This fork packages BlenderGIS as a modern Blender extension with all its Python dependencies bundled, and adds LiDAR, QGIS project and GDAL vector import.

Note : Since 2022, the OpenTopography web service requires an API key. Please register to opentopography.org and request a key. This service is still free.


[Wiki](https://github.com/domlysz/BlenderGIS/wiki/Home) - [FAQ](https://github.com/domlysz/BlenderGIS/wiki/FAQ) - [Quick start guide](https://github.com/domlysz/BlenderGIS/wiki/Quick-start) - [Flowchart](https://raw.githubusercontent.com/wiki/domlysz/blenderGIS/flowchart.jpg)

## Installation

1. Download the zip for your platform from the [latest release](https://github.com/mnmly/BlenderGIS/releases/latest):

   | Platform | File |
   |---|---|
   | Windows | `blendergis-<version>-windows_x64.zip` |
   | macOS, Apple Silicon | `blendergis-<version>-macos_arm64.zip` |
   | macOS, Intel | `blendergis-<version>-macos_x64.zip` |
   | Linux | `blendergis-<version>-linux_x64.zip` |

2. In Blender, open *Edit > Preferences > Get Extensions*, click the *⌄* menu at the top right and choose *Install from Disk...*, then select the zip. Dragging the zip onto the Blender window works too.
3. The tools are in the 3D view: the *GIS* menu of the header, and the *GIS* tab of the sidebar (*N*).

Notes :
- Do not unzip the file, and do not install it as a legacy add-on (*Add-ons > Install*). If an older BlenderGIS add-on is installed, disable and remove it first.
- The zip contains everything needed (laspy, lazrs, pyproj, Pillow, pyogrio with GDAL): no pip install or system GDAL.
- Web features (basemaps, OpenStreetMap, SRTM, XYZ layers of QGIS projects) need *Allow Online Access* in *Preferences > System > Network*.
- To update, install the newer zip the same way. On Windows, restart Blender after updating or disabling the extension, before enabling it again.

## What's new in this fork

**LiDAR import (.las / .laz) :** *GIS > Import > LIDAR (.las .laz)*. The CRS is read from the file (compound CRS are reduced to their horizontal part) or set by hand, points can be clipped to the bounding box of an object, and RGB is kept.

**Point Cloud panel :** in the *GIS* tab of the sidebar, choose what colors the points: RGB, elevation, intensity, classification (ASPRS palette) or any other attribute, with a color ramp and value range. Point size is set here too.

**Colorize points from a raster :** *GIS > Object > Colorize points from raster* (also in the Point Cloud panel), from a raster already in the scene (e.g. an orthophoto layer of a QGIS project) or a GeoTIFF file. *Live* projects the image through the material, *Bake* copies the colors into the points.

**QGIS project import (.qgs / .qgz) :** *GIS > Import > QGIS project (.qgs .qgz)*. The project's layer tree is listed with what can be imported; tick the layers to bring in.
- The scene takes the project CRS, centered on the QGIS map view, and the QGIS groups become collections.
- Layer CRS, single symbol colors and drawing order are kept.
- Supported layers : shapefiles, GeoJSON and other GDAL vector formats, georeferenced rasters, ASCII grids, point clouds and XYZ tile basemaps (downloaded for the map view, at the *Basemap size* resolution).
- The project file is read again on each import, so edits made in QGIS are picked up.

**GDAL vector formats :** *GIS > Import > Vector, GDAL* reads GeoPackage, KML, MapInfo, FlatGeobuf, DXF, CSV, GML... with layer choice, CRS, extrusion field and separate objects.

**Point markers :** point layers (GeoJSON, GDAL formats, shapefiles, QGIS projects) are shown as spheres with a *Point Markers* geometry nodes modifier. Its radius, detail and material can be changed in the modifier; the points themselves stay editable.

**Offline CRS search :** the CRS list of the preferences is searched with pyproj, no web service needed.

## Building the extension
Dependencies are bundled as wheels: laspy and lazrs (LIDAR), pyproj (reprojection), Pillow (images, basemap tiles) and pyogrio (GDAL based vector formats). Download them, then build one zip per platform:

```
uv run --with pip tools/fetch_wheels.py
blender --command extension build --split-platforms
```

Unit tests of the parts that do not need Blender (QGIS project reader, WKB decoder) :

```
uv run --with pytest pytest tests/
```

Logs and the default basemaps cache are stored in the extension user folder (`extensions/.user/<repo>/blendergis`), previously `~/.bgis`.

![las-usage](https://user-images.githubusercontent.com/317202/221855574-bf90814f-0a9a-40a6-8347-d2de6a48e6a5.gif)

--------------------

## Functionalities overview

**GIS datafile import :** Import in Blender most commons GIS data format : Shapefile vector, raster image, geotiff DEM, OpenStreetMap xml.

There are a lot of possibilities to create a 3D terrain from geographic data with BlenderGIS, check the [Flowchart](https://raw.githubusercontent.com/wiki/domlysz/blenderGIS/flowchart.jpg) to have an overview.

Exemple : import vector contour lines, create faces by triangulation and put a topographic raster texture.

![](https://raw.githubusercontent.com/wiki/domlysz/blenderGIS/Blender28x/gif/bgis_demo_delaunay.gif)

**Grab geodata directly from the web :** display dynamics web maps inside Blender 3d view, requests for OpenStreetMap data (buildings, roads ...), get true elevation data from the NASA SRTM mission.

![](https://raw.githubusercontent.com/wiki/domlysz/blenderGIS/Blender28x/gif/bgis_demo_webdata.gif)

**And more :** Manage georeferencing informations of a scene, compute a terrain mesh by Delaunay triangulation, drop objects on a terrain mesh, make terrain analysis using shader nodes, setup new cameras from geotagged photos, setup a camera to render with Blender a new georeferenced raster.
