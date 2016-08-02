# povipy
Bare-bones in-core point cloud viewer made in python. Features:

* basic navigation by translation and rotation (with arcball) and scaling of dataset
* several point rendering methods (from simple OpenGL points to splats)
* load multiple layers at once
* control clipping planes
* colormaps
* runs on windows and mac (linux should work too but not tested)

It was originally developed for visualisation of Medial Axis point clouds (see [skel3d](https://github.com/tudelft3d/skel3d)).

## Installing
First install PyQT5.

Then clone this repository and install with `setup.py`, i.e.
```
python setup.py install
```

## Using
See the example script.
```
python examples/example.py
```

## Limitations
No mechanism to properly handle very big datasets (larger than ~10 million points).
