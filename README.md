Blender GIS
==========
Supported Blender versions : 4.3 to 5.x, on Windows, macOS and Linux, installed as an extension.

This fork packages BlenderGIS as a modern Blender extension with all its Python dependencies bundled, and adds LiDAR, QGIS project and GDAL vector import.

Note : Since 2022, the OpenTopography web service requires an API key. Please register to opentopography.org and request a key. This service is still free.


[Wiki](https://github.com/domlysz/BlenderGIS/wiki/Home) - [FAQ](https://github.com/domlysz/BlenderGIS/wiki/FAQ) - [Quick start guide](https://github.com/domlysz/BlenderGIS/wiki/Quick-start) - [Flowchart](https://raw.githubusercontent.com/wiki/domlysz/blenderGIS/flowchart.jpg)

## Installation

Requires Blender 4.3 or newer. A step by step guide with screenshots, from installing to importing a QGIS project, is in the [tutorial](docs/tutorial.md).

**Recommended: add the BlenderGIS repository**, Blender then installs the right build for your platform and shows updates:

1. In Blender, open *Edit > Preferences*, tick *Allow Online Access* in *System > Network*.
2. In *Get Extensions*, click *Repositories* at the top right, then *+ > Add Remote Repository*.
3. Paste this URL, tick *Check for Updates on Startup* and click *Create*:

   ```
   https://github.com/mnmly/BlenderGIS/releases/latest/download/index.json
   ```

4. Search *BlenderGIS* in *Get Extensions* and click *Install*.
5. The tools are in the 3D view: the *GIS* menu of the header, and the *GIS* tab of the sidebar (*N*).

New releases then appear in *Get Extensions* with an *Update* button. **Restart Blender after updating**: until then the previous pyproj and GDAL libraries stay loaded and reprojections fail.

**Or install a zip by hand:** download the zip for your platform from the [latest release](https://github.com/mnmly/BlenderGIS/releases/latest):

| Platform | File |
|---|---|
| Windows | `blendergis-<version>-windows_x64.zip` |
| macOS, Apple Silicon | `blendergis-<version>-macos_arm64.zip` |
| macOS, Intel | `blendergis-<version>-macos_x64.zip` |
| Linux | `blendergis-<version>-linux_x64.zip` |

then in *Get Extensions*, choose *Install from Disk...* in the *⌄* menu and select the zip (dragging the zip onto the Blender window works too). Zips installed this way are not updated: install the newer zip the same way.

Notes :
- Do not unzip the file, and do not install it as a legacy add-on (*Add-ons > Install*). If an older BlenderGIS add-on is installed, disable and remove it first.
- Switching from a zip installed by hand to the repository: uninstall BlenderGIS first, restart Blender, then install it from the repository.
- The zip contains everything needed (laspy, lazrs, pyproj, Pillow, pyogrio with GDAL): no pip install or system GDAL.
- Web features (basemaps, OpenStreetMap, SRTM, XYZ layers of QGIS projects) and the repository need *Allow Online Access* in *Preferences > System > Network*.
- Basemaps of QGIS projects are saved as images in a *basemaps* folder next to the `.blend` file (copied there when the file is saved): keep it with the `.blend` file.
- Restart Blender after updating, or after disabling the extension before enabling it again (all platforms).

## What's new in this fork

**LiDAR import (.las / .laz) :** *GIS > Import > LIDAR (.las .laz)*. The CRS is read from the file (compound CRS are reduced to their horizontal part) or set by hand, points can be clipped to the bounding box of an object, and RGB is kept.

**Point Cloud panel :** in the *GIS* tab of the sidebar, choose what colors the points: RGB, elevation, intensity, classification (ASPRS palette) or any other attribute, with a color ramp and value range. Point size is set here too.

**Clip and Section panel :** for any point cloud (BlenderGIS imports, PLY or gaussian splat meshes made of vertices, Blender point cloud objects), *Add clip and section* adds a modifier first in the stack. *Clip by* removes the points inside a closed mesh (or keeps only them with *Keep inside*), *Section by* shows only the points within *Thickness* of an object's XY plane. *Add section helper* adds an empty parented to the point cloud that drives the section: rotate or move it to sweep the section through the cloud.

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
