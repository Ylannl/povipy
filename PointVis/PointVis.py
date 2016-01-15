# TODO: clipping planes fixed to data coordinates
# TODO: different LODs/decimations

# TODO: point picking. render point coordinates to texture... 
# TODO: point selection
# TODO: point attribute querying
# TODO: colormap atribute data. using texture? or simple algebra?
# TODO: display normals
# TODO: display other geometries, medial balls, lines to nn's etc

# TODO: structure code
# TODO: make it work with latest vispy
# TODO: proper depth on HUD

# FIXED: LAS reader
# FIXED: layer system for different datasets 
# FIXED: switch between different pointrendering shaders

import numpy as np
import math
from time import time

import OpenGL.GL as gl
import glfw
from transforms import perspective, ortho, scale, translate, rotate, xrotate, yrotate, zrotate

from linalg import quaternion as q
from shader import *#PointShaderProgram, BallShaderProgram


hud_data = np.zeros( 4, [('a_position', np.float32, 2)] )
hud_data['a_position'] = np.array([[-1, 0], [1,0], [0,-1], [0,1]])

cross_vert = """
#version 120

// Attributes
// ------------------------------------
attribute vec2  a_position;

// Varyings
// ------------------------------------
//

void main (void) {
    gl_Position =  vec4(a_position, 0.0, 1.0);
}
"""

cross_frag = """
#version 120

// Main
// ------------------------------------
void main()
{
    gl_FragColor = vec4(0.5,0.5,0.5,1.0);
}
"""

# ------------------------------------------------------------ Canvas class ---

class PointVis():

    def __init__(self, call_func=None, mv=None):
        size = 720, 720

        # Initialize the library
        if not glfw.init():
            return
        # Create a windowed mode window and its OpenGL context
        # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(size[0], size[1], "PointVis", None, None)
        if not self.window:
            glfw.terminate()
            return

        # Make the window's context current
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # glfw.set_window_refresh_callback(self.window, self.on_draw)
        glfw.set_framebuffer_size_callback(self.window, self.on_resize)
        glfw.set_key_callback(self.window, self.on_key_press)
        # glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
        glfw.set_scroll_callback(self.window, self.on_mouse_wheel)
        glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)
        # glfw.set_window_close_callback(self.window, self._on_close)

        # self.hud_program = gloo.Program(cross_vert, cross_frag)
        # self.hud_program.bind(gloo.VertexBuffer(hud_data))

        self.default_view = np.eye(4, dtype=np.float32)
        self.view = self.default_view
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.data_programs = []

        self.current_data_program = 0

        self.rotation = q.quaternion()
        self.scale = 0.6
        self.translation = np.zeros(3)

        self.radius = 0.5 * min(*size)
        self.fov = 5.
        self.camera_position = -12.
        self.near_clip = 1.
        self.far_clip = 100.

        self.draw_hud = False

        self.projection_mode = 'perspective' # 'orthographic'

        # app.Canvas.__init__(self, keys='interactive')
        self.size = size
        self.title = 'PointVis'
        self.bg_white = False
        self.mv = mv
        self.viewpoint_dict = {}
        self.call_func = call_func

    def on_initialize(self):
        self.set_bg()
        gl.glDepthFunc(gl.GL_LESS)

        view_width, view_height = map( lambda x:x/self.radius, self.size )

        translate(self.model, -self.data_center[0], -self.data_center[1], -self.data_center[2])
        for program in self.data_programs:
            program.setUniform('u_model', self.model)

        self.modelscale = .6* 2*min(2.*view_width/self.data_width, 2.*view_height/self.data_height)
        self.scale = self.modelscale
        self.update_view_matrix()

        self.last_mouse_pos = 0,0

        self.on_resize(self.window, *self.size)

    def run(self):
        # Loop until the user closes the window
        while not glfw.window_should_close(self.window):
            # Render here, e.g. using pyOpenGL
            self.on_draw()
            # Swap front and back buffers
            glfw.swap_buffers(self.window)
            # Poll for and process events
            glfw.wait_events()
        glfw.terminate()

    def add_data_source(self, opts, points, normals=None, radii=None, intensity=None, zrange=None):
        # points = points[~np.isnan(points).any(axis=1)]
        m,n = points.shape

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

        data = np.zeros( m, attribute_definitions )
        for definition, di in zip(attribute_definitions, data_list):
            data[definition[0]]=di

        min_xy = np.nanmin( data['a_position'], axis=0 )
        max_xy = np.nanmax( data['a_position'], axis=0 )
        # print min_xy
        # print max_xy
        if len(self.data_programs) == 0:
            self.data_width = max_xy[0] - min_xy[0]
            self.data_height = max_xy[1] - min_xy[1]
            self.data_depth = max_xy[2] - min_xy[2]

            self.data_center = min_xy[0] + self.data_width/2, min_xy[1] + self.data_height/2, min_xy[2] + self.data_depth/2
            # self.data_center = 0,0,0

        if zrange is not None:
             zmin, zmax = zrange
        else:
            zmin, zmax = min_xy[2], max_xy[2]
        # for mode in modes:
        program = PointShaderProgram(options=opts, zrange=(zmin, zmax))
        program.setUniform('u_model', self.model)
        program.setUniform('u_view', self.view)
        program.setUniform('u_projection', self.projection)
        if program.draw_type == gl.GL_POINTS:
            program.setUniform('u_screen_width', self.size[0])
        program.data = data
        self.data_programs.append( program )

    def add_data_source_line(self, coords_start, coords_end, color=(1,0,0)):
        #interleave coordinates
        m,n = coords_start.shape
        vertices = np.empty((m*2,n), dtype=coords_start.dtype)
        vertices[0::2] = coords_start
        vertices[1::2] = coords_end
      
        data = np.empty( 2*m, [('a_position', np.float32, 3)] )
        data['a_position'] = vertices

        program = LineShaderProgram(color)
        program.setUniform('u_model', self.model)
        program.setUniform('u_view', self.view)
        program.setUniform('u_projection', self.projection)
        program.data = data
        self.data_programs.append( program )       

    def add_data_source_ball(self, points, radii, color=(0,1,0)):
        program = BallShaderProgram(points, radii, color)
        program['u_model'] = self.model
        program['u_view'] = self.view
        program['u_projection'] = self.projection
        self.data_programs.append( program )

    def update_view_matrix(self):
        self.view = np.eye(4, dtype=np.float32)
        translate(self.view, self.translation[0], self.translation[1], self.translation[2] )
        scale(self.view, self.scale, self.scale, self.scale)
        self.view = self.view.dot( np.array(q.matrix(self.rotation), dtype=np.float32) )
        # translate(self.view, -self.translation[0], -self.translation[1], -self.translation[2] )
        translate(self.view, 0,0, self.camera_position)
        for program in self.data_programs:
            program.setUniform('u_view', self.view)
            if program.draw_type == gl.GL_POINTS:
                program.setUniform('u_model_scale', self.scale)

    def update_projection_matrix(self):
        view_width, view_height = map( lambda x:x/self.radius, self.size )

        for program in self.data_programs:
            if program.draw_type==gl.GL_POINTS:
                program.setUniform('u_screen_width', min(view_width, view_height))

        if self.projection_mode == 'orthographic':
            self.projection = ortho(-view_width, view_width, -view_height, view_height, self.near_clip, self.far_clip)
        elif self.projection_mode == 'perspective':
            self.projection = perspective(self.fov, view_width / float(view_height), self.near_clip, self.far_clip)

        for program in self.data_programs:
            program.setUniform('u_projection', self.projection)

    def screen2view(self, x,y):
        width, height = self.size
        # print (x-width/2.)/self.radius, ((height-y)-height/2.)/self.radius
        return (x-width/2.)/self.radius, ((height-y)-height/2.)/self.radius

    def set_bg(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        if self.bg_white:
            gl.glClearColor(1,1,1,1)
            # gloo.set_state('translucent', clear_color=np.array([1,1,1,1]) )
        else:
            gl.glClearColor(0.15,0.15,0.15,1)
            # gloo.set_state('translucent', clear_color=np.array([0.15,0.15,0.15,1]) )

    def on_key_press(self, window, key, scancode, action, mods):
        # import ipdb; ipdb.set_trace()
        # if action == glfw.PRESS: print key, self.data_programs[0]['u_point_size']
        if key == glfw.KEY_R and action == glfw.PRESS:
            self.view = np.eye(4, dtype=np.float32)
            self.rotation = q.quaternion()
            self.scale = self.modelscale
            self.translation = np.zeros(3)
            self.camera_position = -12.
            self.near_clip = 2.
            self.far_clip = 100.
            self.update_view_matrix()
            self.update_projection_matrix()
        elif key == glfw.KEY_MINUS and action == glfw.PRESS:
            for program in self.data_programs:
                if program.is_visible and program.draw_type == gl.GL_POINTS:
                    program.setUniform('u_point_size', program.uniforms['u_point_size']/1.2)
        elif key == glfw.KEY_EQUAL and action == glfw.PRESS:
            for program in self.data_programs:
                if program.is_visible and program.draw_type == gl.GL_POINTS:
                    program.setUniform('u_point_size', program.uniforms['u_point_size']*1.2)
        elif key == glfw.KEY_T and action == glfw.PRESS:
            self.rotation = q.quaternion()
            self.update_view_matrix()
        elif key == glfw.KEY_P and action == glfw.PRESS:
            if self.projection_mode == 'perspective':
                self.projection_mode = 'orthographic'
            else:
                self.projection_mode = 'perspective'
            self.update_projection_matrix()
        elif key == glfw.KEY_L and action == glfw.PRESS:
            self.bg_white = not self.bg_white
            self.set_bg()
        elif key == glfw.KEY_V and action == glfw.PRESS:
            self.capture_viewpoint_params()
        elif key == glfw.KEY_LEFT_BRACKET and action == glfw.PRESS:
            self.current_data_program = (self.current_data_program-1) % len(self.data_programs)
        elif key == glfw.KEY_RIGHT_BRACKET:
            self.current_data_program = (self.current_data_program+1) % len(self.data_programs)
        elif glfw.KEY_0 <= key < glfw.KEY_9 and action == glfw.PRESS:
            i = int(chr(key))-1
            if i < len(self.data_programs):
                self.data_programs[i].toggle_visibility()
        self.update()

    def on_resize(self, window, size_x, size_y):
        gl.glViewport(int(0), int(0), int(size_x), int(size_y))

        self.radius = 0.5 * min(size_x, size_y)
        self.size = size_x, size_y

        self.update_projection_matrix()  

    def on_mouse_wheel(self, window, offset_x, offset_y):
        ticks = offset_y
        # x,y = event.pos
        # x -= self.size[0]/2
        # y += self.size[1]/2
        if glfw.get_key(self.window, glfw.KEY_Z) and glfw.get_key(self.window, glfw.KEY_A):
            print 'farnear'
            if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT):
                ticks /= 30
            self.camera_position -= ticks
            self.update_view_matrix()
        elif glfw.get_key(self.window, glfw.KEY_Z):
            if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT):
                ticks /= 30
            new = max(0.1,self.near_clip - ticks)
            if new <= self.far_clip:
                self.near_clip = new
                self.update_projection_matrix()
        elif glfw.get_key(self.window, glfw.KEY_A):
            if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT):
                ticks /= 30
            new = min(1000,self.far_clip - ticks)
            if new >= self.near_clip:
                self.far_clip = new
                self.update_projection_matrix()
        elif glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT):
            if self.projection_mode == 'perspective':
                old_fov = self.fov
                # do `dolly zooming` so that world appears at same size after canging fov
                self.fov = max(5.,self.fov + ticks)
                self.fov = min(120.,self.fov)
                self.camera_position = (self.camera_position * math.tan(math.radians(old_fov)/2.)) / (math.tan(math.radians(self.fov)/2.))
                self.update_projection_matrix()
                self.update_view_matrix()
        else:
            self.scale *= ticks/10 + 1.
            # self.camera_position += ticks/10
            self.update_view_matrix()
        
        self.update()

    def on_mouse_move(self, window, pos_x, pos_y):
        if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT):
            x0,y0 = self.last_mouse_pos
            x1,y1 = pos_x, pos_y
            dx, dy = (x1-x0), (y1-y0)
            #scale to zero plane in projection frustrum
            if self.projection_mode == 'perspective':
                scale = -self.camera_position * math.tan(math.radians(self.fov/2.))
                dx, dy = scale*dx, scale*dy
                #multiply with inverse view matrix and apply translation in world coordinates
                self.translation += np.array([dx/self.radius, -dy/self.radius, 0., 0.]).dot( np.linalg.inv(self.view)) [:3]
            elif self.projection_mode == 'orthographic':
                # this is not fully correct
                self.translation += self.modelscale * np.array([dx, -dy, 0., 0.]).dot( np.linalg.inv(self.view)) [:3]
            
            self.draw_hud = True
        elif glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT):
            x0,y0 = self.screen2view(*self.last_mouse_pos)
            x1,y1 = self.screen2view(pos_x, pos_y)

            v0 = q.arcball(x0, y0)
            v1 = q.arcball(x1, y1)

            self.rotation = q.product(v1, v0, self.rotation)

            self.draw_hud = True
        else:
            self.draw_hud = False
        self.update_view_matrix()
        self.update()

        self.last_mouse_pos = pos_x, pos_y

    def update(self):
        self.on_draw()

    def on_draw(self):
        bits = 0
        bits |= gl.GL_COLOR_BUFFER_BIT
        bits |= gl.GL_DEPTH_BUFFER_BIT
        bits |= gl.GL_STENCIL_BUFFER_BIT
        gl.glClear(bits)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        
        for program in self.data_programs:
            if program.do_blending:
                if self.bg_white:
                    gl.glEnable(gl.GL_BLEND)
                    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_SRC_ALPHA)
                else:
                    gl.glEnable(gl.GL_BLEND)
                    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
            else:
                gl.glDisable(gl.GL_BLEND)
            program.draw()

            # if program.draw_type == 'triangles':
                # gloo.set_state(color_mask=(0,0,0,0))
                # program.draw(program.draw_type, program.indexbuffer)
                # gloo.set_state(color_mask=(1,1,1,1))
        # self.data_programs[self.current_data_program].draw('points')
        # self.data_programs[0].draw('points')
        # self.data_programs[1].draw('points')
        
        # if self.draw_hud:
        #     self.hud_program.draw('lines')
        
        


# if __name__ == '__main__':
    # c = PointVis()
    
    # t1=time()
    # # datadict = read_ply(INFILE)
    # datadict = io_npy.read_npy(INFILE, ['coords', 'normals', 'lfs', 'ma_coords_in', 'ma_radii_in', 'ma_coords_out', 'ma_radii_out'])
    # # datadict = read_xyz('/Users/ravi/project/covadem/TUDelft-OTB/out/out_6.xyz_.xyz')
    # t2=time()
    # print "data loaded in {} s".format(t2-t1)

    # c.add_data_source(
    #     opts=(['with_normals', 'with_point_radius', 'splat_disk'], ['with_normals', 'splat_disk']),
    #     points=datadict['coords'], normals=datadict['normals'], radii=datadict['lfs'])
    # c.add_data_source(
    #     opts = (['splat_point', 'blend'],),
    #     points=datadict['ma_coords_in'])
    # c.add_data_source(
    #     opts = (['splat_point', 'blend'],),
    #     points=datadict['ma_coords_out'])

    # c.show()
    # app.run()
