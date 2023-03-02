Blender GIS
==========
Blender minimum version required : v3.4+

Note : Since 2022, the OpenTopography web service requires an API key. Please register to opentopography.org and request a key. This service is still free.


[Wiki](https://github.com/domlysz/BlenderGIS/wiki/Home) - [FAQ](https://github.com/domlysz/BlenderGIS/wiki/FAQ) - [Quick start guide](https://github.com/domlysz/BlenderGIS/wiki/Quick-start) - [Flowchart](https://raw.githubusercontent.com/wiki/domlysz/blenderGIS/flowchart.jpg)

### Additional installation for LIDAR support
**Change the python path accordingly to your own Blender path**

```
sudo /Applications/Blender.app/Contents/Resources/3.3/python/bin/python3.10 -m pip install laspy laszip pyproj lazrs --target /Applications/Blender.app/Contents/Resources/3.3/python/lib/python3.10/site-packages
```

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
