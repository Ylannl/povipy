from time import time
import sys
from pointio import io_npy
from povi import App
import numpy as np

INFILE = "/Users/ravi/Sync/Phd/subproject/hierarchy/segmentation/data/scan_npy"

if __name__ == '__main__':
    c = App()
    
    if len(sys.argv) > 1:
        INFILE = sys.argv[1]

    t1=time()
    available_keys = io_npy.inspect_npy(INFILE)
    print available_keys

    datadict = io_npy.read_npy(INFILE)
    t2=time()
    print "data loaded in {} s".format(t2-t1)

    mean = np.mean(datadict['coords'], axis=0, dtype=np.float32)
    datadict['coords'] -= mean
    if 'ma_coords_in' in available_keys:
        datadict['ma_coords_in'] -= mean
    if 'ma_coords_out' in available_keys:
        datadict['ma_coords_out'] -= mean

    f1 = datadict['coords']-datadict['ma_coords_in']
    f2 = datadict['coords'][datadict['ma_qidx_in']]-datadict['ma_coords_in']
    ma_bisec = (f1+f2)
    ma_bisec = ma_bisec/np.linalg.norm(ma_bisec, axis=1)[:,None] + datadict['ma_coords_in']
    
    c.add_data_source(
        opts=['splat_disk', 'with_normals'],
        points=datadict['coords'], normals=datadict['normals']
    )

    c.add_data_source(
        opts=['splat_point', 'with_intensity'],
        points=np.concatenate([datadict['ma_coords_in']]), intensity=datadict['ma_segment'].astype(np.float32)
    )
    
    c.add_data_source(
        opts = ['splat_point', 'blend'],
        points=datadict['ma_coords_in']
    )

    c.add_data_source_line(
        coords_start = datadict['ma_coords_in'],
        coords_end = ma_bisec#(f1+f2)+datadict['ma_coords_in']
    )
    
    c.run()