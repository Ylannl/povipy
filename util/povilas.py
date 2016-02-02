from time import time
import sys

import numpy as np
from povi import App
from laspy.file import File
from laspy.util import LaspyException

import click

@click.command(help='OpenGL visualiser for LAS files.')
@click.argument('input', type=click.Path(exists=True))
@click.option('-l', '--limit', default=3.0E6, type=float, help='Limits the number of points on screen.')
def las(input, limit):
    # import ipdb; ipdb.set_trace()
    c = App()
    c.multiview = False
    
    t0=time()
    lasfile = File(input)
    mi, ma = np.array(lasfile.header.min), np.array(lasfile.header.max)    
    count = lasfile.header.count
    offset = mi + (ma - mi)
    offset[2] = 0
    print "{} points loaded from file in {} s".format(count, time()-t0)
    

    filt = np.random.random(count) < (limit / count)
    thin_count = np.sum(filt)
    print "{} points remaining after thinning [factor: {:.2f}]".format(thin_count, float(thin_count)/count)

    points = np.stack((lasfile.get_x_scaled(), lasfile.get_y_scaled(), lasfile.get_z_scaled()), axis=1) - offset
    c.add_data_source(
        opts=['splat_point'],
        points=points[filt],
        colormap='terrain'
    )
    c.data_programs[0].toggle_visibility()

    try:
        intensity = lasfile.get_intensity().astype(np.float32)
        intensity = np.clip(intensity, 0, 600)
        if intensity.max() == 0: raise LaspyException
        c.add_data_source(
            opts=['splat_point', 'with_intensity'],
            points=points[filt],
            intensity=intensity[filt],
            colormap='jet'
        )
    except LaspyException:
        pass

    try:
        classification = lasfile.get_classification()
        if classification.max() == 0: raise LaspyException
        c.add_data_source(
            opts=['splat_point', 'with_intensity'],
            points=points[filt],
            category=classification.astype(np.float32)[filt],
            colormap='las_classes'
        )
    except LaspyException:
        pass
    
    try:
        return_num = lasfile.get_return_num().astype(np.float32)
        if return_num.max() == 1: raise LaspyException
        c.add_data_source(
            opts=['splat_point', 'with_intensity'],
            points=points[filt],
            intensity=return_num[filt],
            colormap='jet'
        )
    except LaspyException:
        pass

    try:
        scan_angle_rank = lasfile.get_scan_angle_rank().astype(np.float32)
        if scan_angle_rank.max() == 0: raise LaspyException
        c.add_data_source(
            opts=['splat_point', 'with_intensity'],
            points=points[filt],
            intensity=scan_angle_rank[filt],
            colormap='jet'
        )
    except LaspyException:
        pass
    
    try:
        gps_time = lasfile.get_gps_time().astype(np.float32)
        if gps_time.max() == 0: raise LaspyException
        gps_time[np.argsort(gps_time)] = np.arange(count)
        c.add_data_source(
            opts=['splat_point', 'with_intensity'],
            points=points[filt],
            intensity=gps_time[filt],
            colormap='jet'
        )
    except LaspyException:
        pass

    c.run()

if __name__ == '__main__':
    las()