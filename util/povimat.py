from time import time
import sys

import numpy as np
from povi import App
from pointio import io_npy

import click

class MAHelper(object):

    def __init__(self, datadict, origin=True):
        if origin==True:
            self.mean = np.mean(datadict['coords'], axis=0, dtype=np.float32)
        else:
            self.mean = 0
        self.D={}
        self.D['coords'] = datadict['coords']-self.mean
        self.D['normals'] = datadict['normals']
        self.D['ma_segment'] = datadict['ma_segment']
        self.m, self.n = self.D['coords'].shape
        self.D['ma_coords_in'] = datadict['ma_coords_in']-self.mean
        self.D['ma_coords_out'] = datadict['ma_coords_out']-self.mean
        self.D['ma_qidx_in'] = datadict['ma_qidx_in']
        self.D['ma_qidx_out'] = datadict['ma_qidx_out']
        self.D['ma_radii_in'] = np.linalg.norm(self.D['coords'] - self.D['ma_coords_in'], axis=1)
        self.D['ma_radii_out'] = np.linalg.norm(self.D['coords'] - self.D['ma_coords_out'], axis=1)

        f1_in = self.D['coords']-self.D['ma_coords_in']
        f2_in = self.D['coords'][self.D['ma_qidx_in']]-self.D['ma_coords_in']
        f1_in = f1_in/np.linalg.norm(f1_in, axis=1)[:,None]
        f2_in = f2_in/np.linalg.norm(f2_in, axis=1)[:,None]
        self.D['ma_bisec_in'] = (f1_in+f2_in)
        self.D['ma_bisec_in'] = self.D['ma_bisec_in']/np.linalg.norm(self.D['ma_bisec_in'], axis=1)[:,None]
        self.D['ma_theta_in'] = np.arccos(np.sum(f1_in*f2_in,axis=1))

        f1_out = self.D['coords']-self.D['ma_coords_out']
        f2_out = self.D['coords'][self.D['ma_qidx_out']]-self.D['ma_coords_out']
        f1_out = f1_out/np.linalg.norm(f1_out, axis=1)[:,None]
        f2_out = f2_out/np.linalg.norm(f2_out, axis=1)[:,None]
        self.D['ma_bisec_out'] = (f1_out+f2_out)
        self.D['ma_bisec_out'] = self.D['ma_bisec_out']/np.linalg.norm(self.D['ma_bisec_out'], axis=1)[:,None]
        self.D['ma_theta_out'] = np.arccos(np.sum(f1_out*f2_out,axis=1))

        self.D['ma_coords'] = np.concatenate([self.D['ma_coords_in'], self.D['ma_coords_out']])
        self.D['ma_bisec'] = np.concatenate([self.D['ma_bisec_in'], self.D['ma_bisec_out']])
        self.D['ma_theta'] = np.concatenate([self.D['ma_theta_in'], self.D['ma_theta_out']])
        self.D['ma_radii'] = np.concatenate([self.D['ma_radii_in'], self.D['ma_radii_out']])


@click.command(help='Visualiser for the MAT')
@click.argument('input', type=click.Path(exists=True))
@click.option('-r', '--max_r', default=190., type=float, help='Only show MAT with a radius lower than specified with this value.')
def mat(input, max_r):
    c = App()
    
    t0=time()
    datadict = io_npy.read_npy(input)
    ma = MAHelper(datadict)
    print("{} points loaded from file in {} s".format(ma.m, time()-t0))

    # import ipdb; ipdb.set_trace()
    c.add_data_source(
        opts=['splat_disk', 'with_normals'],
        points=ma.D['coords'], normals=ma.D['normals']
    )

    if ma.D.has_key('ma_segment'):
        f = np.logical_and(ma.D['ma_radii'] < max_r, ma.D['ma_segment']>0)
        c.add_data_source(
            opts=['splat_point', 'with_intensity'],
            points=ma.D['ma_coords'][f], 
            category=ma.D['ma_segment'][f].astype(np.float32),
            colormap='random'
        )
    
        f = np.logical_and(ma.D['ma_radii'] < max_r, ma.D['ma_segment']==0)
        c.add_data_source(
            opts = ['splat_point', 'blend'],
            points=ma.D['ma_coords'][f]
        )
    else:
        f = ma.D['ma_radii_in'] < max_r
        c.add_data_source(
            opts = ['splat_point', 'blend'],
            points=ma.D['ma_coords_in'][f]
        )
        f = ma.D['ma_radii_out'] < max_r
        c.add_data_source(
            opts = ['splat_point', 'blend'],
            points=ma.D['ma_coords_out'][f]
        )

    f = ma.D['ma_radii'] < max_r
    c.add_data_source_line(
        coords_start = ma.D['ma_coords'][f],
        coords_end = ma.D['ma_bisec'][f]+ma.D['ma_coords'][f]
    )
    c.add_data_source_line(
        coords_start = ma.D['ma_coords'][f],
        coords_end = np.concatenate([ma.D['coords'],ma.D['coords']])[f]
    )
    c.add_data_source_line(
        coords_start = ma.D['ma_coords'][f],
        coords_end = np.concatenate([ma.D['coords'][ma.D['ma_qidx_in']],ma.D['coords'][ma.D['ma_qidx_out']]])[f]
    )
    
    c.run()

if __name__ == '__main__':
    mat()