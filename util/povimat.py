from time import time
import sys
import math
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
        self.D=datadict
        self.D['coords'] = datadict['coords']-self.mean
        self.D['normals'] = datadict['normals']
        if 'ma_segment' in datadict:
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
        self.D['ma_qidx'] = np.concatenate([self.D['ma_qidx_in'], self.D['ma_qidx_out']])


@click.command(help='Visualiser for the MAT')
@click.argument('input', type=click.Path(exists=True))
@click.option('-R', '--max_r', default=190., type=float, help='Only show MAT with a radius lower than specified with this value.')
@click.option('-r', '--min_r', default=0., type=float, help='Only show MAT with a radius higher than specified with this value.')
@click.option('-n', '--near_clip', default=0.1, type=float, help='Near clip value.')
@click.option('-f', '--far_clip', default=100.0, type=float, help='Far clip value.')
def mat(input, min_r, max_r, near_clip, far_clip):
    c = App(near_clip=near_clip, far_clip=far_clip)
    
    t0=time()
    datadict = io_npy.read_npy(input)
    ma = MAHelper(datadict)
    print(("{} points loaded from file in {} s".format(ma.m, time()-t0)))

    # import ipdb; ipdb.set_trace()
    c.add_data_source(
        name = 'Surface points',
        opts=['splat_disk', 'with_normals'],
        points=ma.D['coords'], normals=ma.D['normals']
    )

    if 'decimate_lfs' in ma.D:
        f = ma.D['decimate_lfs']
        c.add_data_source(
            name = 'Surface points lfs',
            opts=['splat_disk', 'with_normals', 'with_intensity'],
            intensity=ma.D['lfs'],
            points=ma.D['coords'], normals=ma.D['normals']
        )
        
        c.add_data_source(
            name = 'Surface points decimated',
            opts=['splat_disk', 'with_normals'],
            points=ma.D['coords'][f], normals=ma.D['normals'][f]
        )

    f_r = np.logical_and(ma.D['ma_radii'] < max_r, ma.D['ma_radii'] > min_r)

    if 'ma_segment' in ma.D:
        f = np.logical_and(f_r, ma.D['ma_segment']>0)
        c.add_data_source(
            name = 'MAT points',
            opts=['splat_point', 'with_intensity'],
            points=ma.D['ma_coords'][f], 
            category=ma.D['ma_segment'][f].astype(np.float32),
            colormap='random'
        )
    
        f = np.logical_and(f_r, ma.D['ma_segment']==0)
        c.add_data_source(
            name = 'MAT points unsegmented',
            opts = ['splat_point', 'blend'],
            points=ma.D['ma_coords'][f]
        )
    else:
        f_ri = np.logical_and(ma.D['ma_radii_in'] < max_r, ma.D['ma_radii_in'] > min_r)
        c.add_data_source(
            name = 'MAT points interior',
            opts = ['splat_point', 'blend'],
            points=ma.D['ma_coords_in'][f_ri]
        )
        f_ro = np.logical_and(ma.D['ma_radii_out'] < max_r, ma.D['ma_radii_out'] > min_r)
        c.add_data_source(
            name = 'MAT points exterior',
            opts = ['splat_point', 'blend'],
            points=ma.D['ma_coords_out'][f_ro]
        )

    
    c.add_data_source_line(
        name = 'Bisectors',
        coords_start = ma.D['ma_coords'][f_r],
        coords_end = ma.D['ma_bisec'][f_r]+ma.D['ma_coords'][f_r]
    )
    c.add_data_source_line(
        name = 'Primary spokes',
        coords_start = ma.D['ma_coords'][f_r],
        coords_end = np.concatenate([ma.D['coords'],ma.D['coords']])[f_r]
    )
    c.add_data_source_line(
        name = 'Secondary spokes',
        coords_start = ma.D['ma_coords'][f_r],
        coords_end = np.concatenate([ma.D['coords'][ma.D['ma_qidx_in']],ma.D['coords'][ma.D['ma_qidx_out']]])[f_r]
    )
    
    c.run()

if __name__ == '__main__':
    mat()