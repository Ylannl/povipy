from time import time
import sys
# from masbpy.io_ply import read_ply
from pointio import io_npy
from povi import App
import numpy as np

# INFILE = "/Users/ravi/git/masbpy/house_dyke_tree_npy"
# INFILE = "/Users/ravi/git/masb/lidar/rdam_blokken_npy_lfsk10"
# INFILE = "/Users/ravi/git/masb/lidar/wijk_hoek van holland/wijk_hoek_van_holland_npy_"
# INFILE = "/Volumes/Data/Data/pointcloud/AHN2_matahn_samples/ringdijk_opmeer_npy"
# INFILE = "/Volumes/Data/Data/pointcloud/AHN2_matahn_samples/lek_bergstoep_npy"
# INFILE = "/Volumes/Data/Data/pointcloud/open_topography/CA_pressure_ridge_npy"

INFILE = "/Users/ravi/Sync/Phd/subproject/hierarchy/segmentation/data/scan_npy"

if __name__ == '__main__':
    c = App()
    ma_min, ma_max = -1000, 1000
    if len(sys.argv) > 1:
        INFILE = sys.argv[1]
    if len(sys.argv) > 3:
        ma_min, ma_max = float(sys.argv[2]), float(sys.argv[3])
    t1=time()
    # datadict = read_ply(INFILE)
    lfsdec = []

    available_keys = io_npy.inspect_npy(INFILE)
    print available_keys

    # keys = ['coords', 'normals', 'ma_coords_in', 'ma_coords_out', 'ma_coords_in_median_dist', 'ma_coords_out_median_dist']+lfsdec
    # if 'lfs' in available_keys:
    #     keys.append('lfs')

    # keys = ['coords', 'normals', 'lfs', 'lfs_k10', 'lfs_k20', 'lfs_filter', 'ma_coords_in', 'ma_radii_in', 'ma_coords_out', 'ma_radii_out', 'surfvar', 'meandist_in', 'meandist_out', 'planefitcrit']

    datadict = io_npy.read_npy(INFILE)
    # datadict_nd = io_npy.read_npy(INFILE_nd, ['coords', 'normals', 'lfs', 'ma_coords_in', 'ma_coords_out'])
    # datadict = io_npy.read_npy(INFILE, ['coords', 'normals', 'lfs', 'ma_coords_in', 'ma_radii_in', 'ma_coords_out', 'ma_radii_out', 'theta_out']+lfsdec)
    # datadict = read_xyz('/Users/ravi/project/covadem/TUDelft-OTB/out/out_6.xyz_.xyz')
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
        # opts=(['with_normals', 'with_point_radius', 'splat_disk'], ['with_normals', 'splat_disk']),
        opts=['splat_disk', 'with_normals'],
        # opts=(['with_normals'],['with_normals', 'splat_disk'],['with_normals', 'splat_disk', 'with_point_radius'],),
        points=datadict['coords'], normals=datadict['normals'])
        # points=datadict['coords'], normals=datadict['normals'], radii=datadict['lfs'])

    c.add_data_source(
        opts=['adaptive_point', 'with_intensity'],
        points=np.concatenate([datadict['ma_coords_in']]), intensity=datadict['ma_segment'].astype(np.float32))

    if 'ma_coords_in' in available_keys:
        c.add_data_source(
            opts = ['adaptive_point', 'blend'],
            # opts = 'blend'
            points=datadict['ma_coords_in']
        )

    if 'ma_coords_out' in available_keys:
        c.add_data_source(
            opts = ['adaptive_point', 'blend'],
            # opts = 'blend'
            points=datadict['ma_coords_out']
        )

    c.add_data_source_line(
        coords_start = datadict['ma_coords_in'],
        coords_end = ma_bisec#(f1+f2)+datadict['ma_coords_in']
    )
    c.add_data_source_line(
        coords_start = datadict['ma_coords_in'],
        coords_end = datadict['coords']
    )
    c.add_data_source_line(
        coords_start = datadict['coords'][datadict['ma_qidx_in']],
        coords_end = datadict['ma_coords_in']
    )
    c.add_data_source_line(
        coords_start = datadict['ma_coords_out'],
        coords_end = datadict['coords']
    )
    c.add_data_source_line(
        coords_start = datadict['coords'][datadict['ma_qidx_out']],
        coords_end = datadict['ma_coords_out']
    )

    
    c.on_initialize()
    c.run()