# povipy
Bare-bones in-core point cloud viewer made in python. Features:

* basic navigation by translation and rotation (with arcball) and scaling of dataset
* several point rendering methods (from simple OpenGL points to splats)
* load multiple layers at once
* control clipping planes
* colormaps

## Installing
First install [GLFW](http://www.glfw.org) 3. With Mac/Homebrew this is easy:
```
brew install glfw3
```
for other platforms check the [GLFW download page](http://www.glfw.org/download.html).

Then install povipy with pip.
```
pip install git+https://github.com/Ylannl/povipy.git
```

## Using
For LAS files:
```
povilas.py mydata.las
```
LAZ is also possible if [laspy](https://github.com/grantbrown/laspy) is able to detect `laszip` on your system.

For MAT datasets created with [masbcpp](https://github.com/tudelft3d/masbcpp):
```
povimat.py mydata_npy
```

## Limitations
No mechanism to properly handle very big datasets (larger than ~10 million points).
