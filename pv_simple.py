from time import time
import sys
from pointio import io_npy
from povi import App
import numpy as np

INFILE = "/Users/ravi/Sync/Phd/subproject/hierarchy/segmentation/data/scan_npy"

class MAHelper(object):

    def __init__(self, datadict, origin=True):
        if origin==True:
            self.mean = np.mean(datadict['coords'], axis=0, dtype=np.float32)
        else:
            self.mean = 0
        self.coords = datadict['coords']-self.mean
        self.normals = datadict['normals']
        self.ma_segment = datadict['ma_segment']
        self.m, self.n = self.coords.shape
        self.ma_coords_in = datadict['ma_coords_in']-self.mean
        self.ma_coords_out = datadict['ma_coords_out']-self.mean
        self.ma_qidx_in = datadict['ma_qidx_in']
        self.ma_qidx_out = datadict['ma_qidx_out']
        self.ma_radii_in = np.linalg.norm(self.coords - self.ma_coords_in, axis=1)
        self.ma_radii_out = np.linalg.norm(self.coords - self.ma_coords_out, axis=1)

        f1_in = self.coords-self.ma_coords_in
        f2_in = self.coords[self.ma_qidx_in]-self.ma_coords_in
        f1_in = f1_in/np.linalg.norm(f1_in, axis=1)[:,None]
        f2_in = f2_in/np.linalg.norm(f2_in, axis=1)[:,None]
        self.ma_bisec_in = (f1_in+f2_in)
        self.ma_bisec_in = self.ma_bisec_in/np.linalg.norm(self.ma_bisec_in, axis=1)[:,None]
        self.ma_theta_in = np.arccos(np.sum(f1_in*f2_in,axis=1))

        f1_out = self.coords-self.ma_coords_out
        f2_out = self.coords[self.ma_qidx_out]-self.ma_coords_out
        f1_out = f1_out/np.linalg.norm(f1_out, axis=1)[:,None]
        f2_out = f2_out/np.linalg.norm(f2_out, axis=1)[:,None]
        self.ma_bisec_out = (f1_out+f2_out)
        self.ma_bisec_out = self.ma_bisec_out/np.linalg.norm(self.ma_bisec_out, axis=1)[:,None]
        self.ma_theta_out = np.arccos(np.sum(f1_out*f2_out,axis=1))

        self.ma_coords = np.concatenate([self.ma_coords_in, self.ma_coords_out])
        self.ma_bisec = np.concatenate([self.ma_bisec_in, self.ma_bisec_out])
        self.ma_theta = np.concatenate([self.ma_theta_in, self.ma_theta_out])
        self.ma_radii = np.concatenate([self.ma_radii_in, self.ma_radii_out])

if __name__ == '__main__':
    c = App()
    
    if len(sys.argv) > 1:
        INFILE = sys.argv[1]

    t1=time()
    available_keys = io_npy.inspect_npy(INFILE)
    print available_keys

    datadict = io_npy.read_npy(INFILE)
    ma = MAHelper(datadict)
    # filt = ma.ma_radii < 190.
    
    filt = np.logical_and(ma.ma_radii < 190., ma.ma_segment>0)
    # filt = ma.ma_segment>=0
    # filt = ma.ma_segment>0
    t2=time()
    print "data loaded in {} s".format(t2-t1)
    # import ipdb; ipdb.set_trace()
    c.add_data_source(
        opts=['splat_disk', 'with_normals'],
        points=ma.coords, normals=ma.normals
    )

    c.add_data_source(
        opts=['splat_point', 'with_intensity'],
        points=ma.ma_coords[filt], 
        category=ma.ma_segment[filt].astype(np.float32),
        colormap='random'
    )
    
    f = np.logical_and(ma.ma_radii < 190., ma.ma_segment==0)
    c.add_data_source(
        opts = ['splat_point', 'blend'],
        points=ma.ma_coords[f]
    )

    c.add_data_source_line(
        coords_start = ma.ma_coords,
        coords_end = ma.ma_bisec+ma.ma_coords
    )
    c.add_data_source_line(
        coords_start = ma.ma_coords,
        coords_end = np.concatenate([ma.coords,ma.coords])
    )
    c.add_data_source_line(
        coords_start = ma.ma_coords,
        coords_end = np.concatenate([ma.coords[ma.ma_qidx_in],ma.coords[ma.ma_qidx_out]])
    )
    
    c.run()