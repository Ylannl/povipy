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

    lasfile = File(INFILE)
    
    offset = np.array(lasfile.header.min)
    points = np.stack((lasfile.get_x_scaled(), lasfile.get_y_scaled(), lasfile.get_z_scaled()), axis=1) - offset
    classification = lasfile.get_classification()
    
    # filt = np.logical_and(ma.ma_radii < 190., ma.ma_segment>0)
    # filt = ma.ma_segment>=0
    # filt = ma.ma_segment>0
    # t2=time()
    # print "data loaded in {} s".format(t2-t1)
    # import ipdb; ipdb.set_trace()
    c.add_data_source(
        opts=['splat_point'],
        points=points,
        colormap='terrain'
    )

    c.add_data_source(
        opts=['splat_point', 'with_intensity'],
        points=points,
        intensity=classification.astype(np.float32),
        colormap='random'
    )

    # c.add_data_source(
    #     opts=['splat_point', 'with_intensity'],
    #     points=ma.ma_coords[filt], 
    #     category=ma.ma_segment[filt].astype(np.float32),
    #     colormap='random'
    # )
    
    # f = np.logical_and(ma.ma_radii < 190., ma.ma_segment==0)
    # c.add_data_source(
    #     opts = ['splat_point', 'blend'],
    #     points=ma.ma_coords[f]
    # )

    
    c.run()