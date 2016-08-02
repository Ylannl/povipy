import numpy as np
import click

from povi import App, Layer

def get_3d_pointcloud():
    n_angles = 36
    n_radii = 8

    # An array of radii
    # Does not include radius r=0, this is to eliminate duplicate points
    radii = np.linspace(0.125, 1.0, n_radii)

    # An array of angles
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

    # Repeat all angles for each radius
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

    # Convert polar (radii, angles) coords to cartesian (x, y) coords
    # (0, 0) is added here. There are no duplicate points in the (x, y) plane
    x = np.append(0, (radii*np.cos(angles)).flatten())
    y = np.append(0, (radii*np.sin(angles)).flatten())

    # Pringle surface
    z = np.sin(-x*y)

    return np.transpose(np.vstack([x,y,z]))

@click.command(help='Example viewing program')
def mat():
    c = App()
    
    example_layer = c.add_layer(Layer(name='Example Layer'))
    example_layer.add_data_source(
        name = 'Points', # this should be a unique name within this layer
        opts=['splat_point','adaptive_point',], # rendering options
        points=get_3d_pointcloud()
    )
    
    c.run()

if __name__ == '__main__':
    mat()