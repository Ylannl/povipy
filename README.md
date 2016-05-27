# povipy
Bare-bones in-core point cloud viewer made in python. Features:

* basic navigation by translation and rotation (with arcball) and scaling of dataset
* several point rendering methods (from simple OpenGL points to splats)
* load multiple layers at once
* control clipping planes
* colormaps
* runs on windows and mac (linux should work too but not tested)

## Installing
First install PyQT5.

Then install povipy with pip.
```
pip install git+https://github.com/Ylannl/povipy.git --process-dependency-links
```

## Using
For MAT datasets created with [masbcpp](https://github.com/tudelft3d/masbcpp):
```
povimat.py mydata_npy
```

## Limitations
No mechanism to properly handle very big datasets (larger than ~10 million points).
