from collections import OrderedDict
import numpy as np
from .shader import *


class LayerManager(object):

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()

    def programs(self, with_names=False):
        for layer in self.layers:
            for name, program in layer.programs.items():
                if with_names:
                    yield name, program
                else:
                    yield program

class Layer(object):

    def __init__(self, name=None):
        self.name = name
        self.is_visible = True
        self.programs = OrderedDict()
        self.bb_min = None
        self.bb_max = None

    def enable(self):
        self.is_visible = True

    def disable(self):
        self.is_visible = False

    def toggle(self):
        self.is_visible = not self.is_visible

    # def add_program(self, name, program):
    #     self.programs[name] = program 

    def draw(self):
        if self.is_visible:
            for p in self.programs.values():
                p.draw()

    def update_box(self, data_array):
        d_min = tuple(np.nanmin( data_array, axis=0 ))
        d_max = tuple(np.nanmax( data_array, axis=0 ))
        if self.bb_min is None:
            self.bb_min = d_min
            self.bb_max = d_max
        else:
            self.bb_min = np.min([d_min, self.bb_min], axis=0)
            self.bb_max = np.max([d_max, self.bb_max], axis=0)
        return d_min, d_max
    
    def get_center(self):
        self.data_range = self.bb_max - self.bb_min
        data_center = self.bb_min + self.data_range/2
        return data_center

    def add_data_source(self, name, opts, points, normals=None, radii=None, intensity=None, category=None, zrange=None, **kwargs):
        # points = points[~np.isnan(points).any(axis=1)]
        m,n = points.shape
        d_min, d_max = self.update_box(points)

        attribute_definitions = []
        data_list = []
        attribute_definitions.append(('a_position', np.float32, 3))
        data_list = [points]
        if normals is not None:
            attribute_definitions.append(('a_normal', np.float32, 3))
            data_list.append(normals)
        if radii is not None:
            attribute_definitions.append(('a_splat_radius', np.float32, 1))
            data_list.append(radii)
        if intensity is not None:
            attribute_definitions.append(('a_intensity', np.float32, 1))
            intensity *= 1./np.nanmax(intensity)
            # import pdb; pdb.set_trace()
            data_list.append(intensity)
        if category is not None:
            attribute_definitions.append(('a_intensity', np.float32, 1))
            intensity = (category%256)/256.
            # import pdb; pdb.set_trace()
            data_list.append(intensity)

        data = np.zeros( m, attribute_definitions )
        for definition, di in zip(attribute_definitions, data_list):
            data[definition[0]]=di

        if zrange is not None:
             zmin, zmax = zrange
        else:
            zmin, zmax = d_min[2], d_max[2]
        # for mode in modes:
        program = PointShaderProgram(options=opts, zrange=(zmin, zmax), **kwargs)
        program.name = name
        program.setAttributes(data)

        self.programs[name] = program

        return program

    def add_data_source_line(self, name, coords_start, coords_end, **args):
        #interleave coordinates
        m,n = coords_start.shape
        self.update_box(coords_start)

        vertices = np.empty((m*2,n), dtype=coords_start.dtype)
        vertices[0::2] = coords_start
        vertices[1::2] = coords_end
      
        data = np.empty( 2*m, [('a_position', np.float32, 3)] )
        data['a_position'] = vertices

        program = LineShaderProgram(**args)

        program.name = name
        program.setAttributes(data)
        self.programs[name] = program

        return program

    def add_data_source_triangle(self, name, coords, normals, **args):
        self.update_box(coords)
        m,n = coords.shape

        data = np.empty( m, [('a_position', np.float32, 3), ('a_normal', np.float32, 3)] )
        data['a_position'] = coords
        data['a_normal'] = normals

        program = TriangleShaderProgram(**args)

        program.name = name
        program.setAttributes(data)
        self.programs[name] = program

        return program

    # def add_data_source_ball(self, name, points, radii, color=(0,1,0)):
    #     program = BallShaderProgram(points, radii, color)
    #     program.name = name
    #     program['u_model'] = self.model
    #     program['u_view'] = self.view
    #     program['u_projection'] = self.projection
    #     self.data_programs[name] = program

    #     return program


class LinkedLayer(Layer):
    def mask(self, mask):
        if mask is None:
            for p in self.programs.values():
                p.updateAttributes(filter=None)
        elif mask.dtype==bool:
            for p in self.programs.values():
                if p.draw_type == 'points':
                    p.updateAttributes(filter=mask)
                elif p.draw_type == 'lines':
                    p.updateAttributes(filter=np.repeat(mask,2))
        