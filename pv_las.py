from time import time
import sys
from povi import App
import numpy as np

from laspy.file import File

INFILE = 'data/ahn3_zoetermeer_stadshart_small.laz'

if __name__ == '__main__':
    c = App()
    
    if len(sys.argv) > 1:
        INFILE = sys.argv[1]

    t1=time()
    lasfile = File(INFILE)
    
    mi, ma = np.array(lasfile.header.min), np.array(lasfile.header.max)    
    offset = mi + (ma - mi)
    # import ipdb; ipdb.set_trace()
    count = lasfile.header.count
    points = np.stack((lasfile.get_x_scaled(), lasfile.get_y_scaled(), lasfile.get_z_scaled()), axis=1) - offset
    classification = lasfile.get_classification()
    
    intensity = lasfile.get_intensity().astype(np.float32)
    intensity = np.clip(intensity, 0, 600)

    gps_time = lasfile.get_gps_time().astype(np.float32)
    gps_time[np.argsort(gps_time)] = np.arange(count)

    scan_angle_rank = lasfile.get_scan_angle_rank().astype(np.float32)
    return_num = lasfile.get_return_num().astype(np.float32)
    
    # filt = np.logical_and(ma.ma_radii < 190., ma.ma_segment>0)
    # filt = ma.ma_segment>=0
    # filt = ma.ma_segment>0
    t2=time()
    print "loaded {} points in {} s".format(count, t2-t1)
    
    c.add_data_source(
        opts=['splat_point'],
        points=points,
        colormap='terrain'
    )

    c.add_data_source(
        opts=['splat_point', 'with_intensity'],
        points=points,
        category=classification.astype(np.float32),
        colormap='las_classes'
    )

    c.add_data_source(
        opts=['splat_point', 'with_intensity'],
        points=points,
        intensity=intensity,
        colormap='jet'
    )

    c.add_data_source(
        opts=['splat_point', 'with_intensity'],
        points=points,
        intensity=gps_time,
        colormap='jet'
    )

    c.add_data_source(
        opts=['splat_point', 'with_intensity'],
        points=points,
        intensity=scan_angle_rank,
        colormap='jet'
    )
    c.add_data_source(
        opts=['splat_point', 'with_intensity'],
        points=points,
        intensity=return_num,
        colormap='jet'
    )

    c.run()