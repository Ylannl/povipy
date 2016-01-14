from time import time
import sys
# from masbpy.io_ply import read_ply
from pointio import io_npy
from PointVis import PointVis
import numpy as np

# INFILE = "/Users/ravi/git/masbpy/house_dyke_tree_npy"
# INFILE = "/Users/ravi/git/masb/lidar/rdam_blokken_npy_lfsk10"
INFILE = "/Users/ravi/git/masb/lidar/wijk_hoek van holland/wijk_hoek_van_holland_npy_"
# INFILE = "/Volumes/Data/Data/pointcloud/AHN2_matahn_samples/ringdijk_opmeer_npy"
# INFILE = "/Volumes/Data/Data/pointcloud/AHN2_matahn_samples/lek_bergstoep_npy"
# INFILE = "/Volumes/Data/Data/pointcloud/open_topography/CA_pressure_ridge_npy"

# INFILE = "/Volumes/Data/Data/pointcloud/open_topography/CA_pressure_ridge_nodenoise_npy"

if __name__ == '__main__':
    c = PointVis()
    ma_min, ma_max = -1000, 1000
    if len(sys.argv) > 1:
        INFILE = sys.argv[1]
    if len(sys.argv) > 3:
        ma_min, ma_max = float(sys.argv[2]), float(sys.argv[3])
    t1=time()
    # datadict = read_ply(INFILE)
    lfsdec = []#['decimate_lfs_0.4_sqFalse']# 
    # lfsdec = ['decimate_lfs_0.4_sqFalse', 'decimate_lfs_filter_0.4_sqFalse']
    # lfsdec = ['decimate_lfs_filter_'+str(e)+'_sqFalse' for e in [0.1,0.2,0.4,0.6,0.9,1.3,1.7,2.0]]#[0.1,0.2,0.4,0.6,0.9,1.3,1.7,2.0]
    # lfsdec = []
    # griddec = ['gridthin_'+str(2**e*0.5) for e in range(4) ]

    available_keys = io_npy.inspect_npy(INFILE)
    print available_keys

    keys = ['coords', 'normals', 'ma_coords_in', 'ma_coords_out', 'ma_coords_in_median_dist', 'ma_coords_out_median_dist']+lfsdec
    if 'lfs' in available_keys:
        keys.append('lfs')

    # keys = ['coords', 'normals', 'lfs', 'lfs_k10', 'lfs_k20', 'lfs_filter', 'ma_coords_in', 'ma_radii_in', 'ma_coords_out', 'ma_radii_out', 'surfvar', 'meandist_in', 'meandist_out', 'planefitcrit']

    datadict = io_npy.read_npy(INFILE, keys)
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

    c.add_data_source(
        # opts=(['with_normals', 'with_point_radius', 'splat_disk'], ['with_normals', 'splat_disk']),
        mode=('fixed_point'),
        # opts=(['with_normals'],['with_normals', 'splat_disk'],['with_normals', 'splat_disk', 'with_point_radius'],),
        points=datadict['coords'], normals=datadict['normals'])
        # points=datadict['coords'], normals=datadict['normals'], radii=datadict['lfs'])
    
    c.on_initialize()
    c.run()