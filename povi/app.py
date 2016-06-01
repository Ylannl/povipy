# TODO: clipping planes fixed to data coordinates
# TODO: different LODs/decimations

# TODO: point selection
# TODO: point attribute querying

# TODO: proper depth on HUD

import numpy as np
import math
import os
from time import time
from collections import OrderedDict

import OpenGL.GL as gl

from PyQt5 import uic
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import (QOpenGLContext, QSurfaceFormat, QWindow)
from PyQt5.QtWidgets import QApplication, QWidget, QToolBox

from .transforms import perspective, ortho, scale, translate, rotate, xrotate, yrotate, zrotate

from .linalg import quaternion as q
from .shader import *

class App(QApplication):

    def __init__(self, args=[], **kwargs):
        # self.app = QApplication([])
        super(App, self).__init__(args)

        self.viewerWindow = ViewerWindow(**kwargs)
        self.dialog = None
        
        self.add_data_source = self.viewerWindow.add_data_source
        self.add_data_source_line = self.viewerWindow.add_data_source_line
        self.data_programs = self.viewerWindow.data_programs
        
        self.viewerWindow.visibility_toggle_listeners.append(self.set_layer_visibility)

    def run(self):
        self.viewerWindow.run()
        if self.dialog is None:
            self.dialog = ToolsDialog(self)
        self.dialog.show()
        self.exec_()
        
    def set_layer_visibility(self, name, is_visible):
        items = self.dialog.ui.listWidget_layers.findItems(name,Qt.MatchExactly)
        for item in items:
            item.setSelected(is_visible)
        
    def set_layer_selection(self):
        selected_names = [item.data(0) for item in self.dialog.ui.listWidget_layers.selectedItems()]
        for name, program in self.viewerWindow.data_programs.items():
            if name in selected_names:
                program.is_visible = True
            elif not name.startswith('graph'):
                program.is_visible = False
        self.viewerWindow.render()

class ToolsDialog(QWidget):
    def __init__(self, app, parent=None):
        super(ToolsDialog, self).__init__(parent)
        self.ui = uic.loadUi(os.path.join(os.path.dirname(__file__),'tools.ui'), self)
        self.app = app

        # populate datalayers list
        # print self.app.viewerWindow.data_programs.keys()
        l=[]
        for program_name in list(self.app.viewerWindow.data_programs.keys()):
            if not program_name.startswith('graph'):
                l.append(program_name)
        self.ui.listWidget_layers.addItems(l)

        # self.ui.doubleSpinBox_filterRadius.valueChanged.connect(self.app.update_radius)
        # self.ui.spinBox_linkcount.valueChanged.connect(self.app.filter_linkcount)
        # self.ui.pushButton_regraph.clicked.connect(self.app.draw_graphs)
        # self.ui.groupBox_component.clicked.connect(self.app.filter_component_all)
        # self.ui.comboBox_component.activated.connect(self.app.filter_component)
        self.ui.listWidget_layers.itemSelectionChanged.connect(self.app.set_layer_selection)

class ViewerWindow(QWindow):

    instructions = """
--Key controls
0-9     - toggle data layers
r       - reset view parameters
c       - clip view
t       - view from top
l       - light/dark background
=       - increase point size
-       - decrease point size

--Mouse controls
drag                 - rotate dataset (arcball)
shift + move         - translate dataset
scroll               - scale dataset
shift + scroll       - change field of view
ctrl + scroll        - move far clipping plane
alt + scroll         - move near clipping plane
ctrl + alt + scroll  - move far and near clipping plane simultaniously
"""

    def __init__(self, parent=None, **kwargs):
        super(ViewerWindow, self).__init__(parent)
        self.setSurfaceType(QWindow.OpenGLSurface)

        format = QSurfaceFormat()
        format.setVersion(3, 3)
        format.setProfile(QSurfaceFormat.CoreProfile)
        format.setStereo(False)
        format.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
        format.setDepthBufferSize(24)
        
        self.context = QOpenGLContext()
        self.context.setFormat(format)
        if not self.context.create():
            raise Exception('self.context.create() failed')
        self.create()

        size = 720, 720
        self.resize(*size)

        self.context.makeCurrent(self)
        self.hud_program = CrossHairProgram()

        self.default_view = np.eye(4, dtype=np.float32)
        self.view = self.default_view
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.data_programs = OrderedDict()
        self.visibility_toggle_listeners = []
        self.multiview = True

        self.rotation = q.quaternion()
        self.scale = 0.6
        self.translation = np.zeros(3)

        self.radius = 0.5 * min(*size)
        self.fov = 5.
        self.camera_position = -12.
        self.near_clip = .1
        if kwargs.has_key('near_clip'):
            self.near_clip = kwargs['near_clip']
        self.far_clip = 100.
        if kwargs.has_key('far_clip'):
            self.far_clip = kwargs['far_clip']
        

        self.projection_mode = 'perspective' # 'orthographic'

        self.size = size
        self.bg_white = False
        self.viewpoint_dict = {}

        print((self.instructions))

    def run(self):
        self.initialize()
        self.show()

    def set_layer_visibility(self, name, is_visible):
        for listener in self.visibility_toggle_listeners:
            listener(name, is_visible)
        self.data_programs[name].is_visible = is_visible

    def center_view(self, center=None):
        if center is None:
            center = self.data_center
        self.translation = np.zeros(3)
        self.model = np.eye(4, dtype=np.float32)
        translate(self.model, -center[0], -center[1], -center[2])
        for program in list(self.data_programs.values()):
            program.setUniform('u_model', self.model)
        self.update_view_matrix()

    def initialize(self):
        self.context.makeCurrent(self)

        gl.glDepthMask(gl.GL_TRUE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)

        self.set_bg()

        view_width, view_height = [x/self.radius for x in self.size]

        translate(self.model, -self.data_center[0], -self.data_center[1], -self.data_center[2])
        for program in list(self.data_programs.values()):
            program.setUniform('u_model', self.model)

        self.modelscale = .6* 2*min(2.*view_width/self.data_width, 2.*view_height/self.data_height)
        self.scale = self.modelscale
        self.update_view_matrix()

        self.last_mouse_pos = 0,0

        self.on_resize(*self.size)

    def render(self):
        if not self.isExposed():
            return
        self.context.makeCurrent(self)

        bits = 0
        bits |= gl.GL_COLOR_BUFFER_BIT
        bits |= gl.GL_DEPTH_BUFFER_BIT
        bits |= gl.GL_STENCIL_BUFFER_BIT
        gl.glClear(bits)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        
        for program in list(self.data_programs.values()):
            if program.do_blending:
                if self.bg_white:
                    gl.glEnable(gl.GL_BLEND)
                    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_SRC_ALPHA)
                else:
                    gl.glEnable(gl.GL_BLEND)
                    gl.glBlendFunc(gl.GL_ONE, gl.GL_SRC_ALPHA)
            else:
                gl.glDisable(gl.GL_BLEND)
            program.draw()
        
        if self.hud_program.is_visible:
            self.hud_program.draw()

        self.context.swapBuffers(self)

    def event(self, event):
        # print event.type()
        if event.type() == QEvent.UpdateRequest:
            self.render()
            return True

        return super(ViewerWindow, self).event(event)

    def exposeEvent(self, event):
        self.render()

    def resizeEvent(self, event):
        size = event.size()
        self.on_resize(size.width(), size.height())
        self.render()

    def add_data_source(self, name, opts, points, normals=None, radii=None, intensity=None, category=None, zrange=None, **kwargs):
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
        if category is not None:
            attribute_definitions.append(('a_intensity', np.float32, 1))
            intensity = (category%256)/256.
            # import pdb; pdb.set_trace()
            data_list.append(intensity)

        data = np.zeros( m, attribute_definitions )
        for definition, di in zip(attribute_definitions, data_list):
            data[definition[0]]=di

        min_xy = np.nanmin( data['a_position'], axis=0 )
        max_xy = np.nanmax( data['a_position'], axis=0 )
        # print min_xy
        # print max_xy
        if len(list(self.data_programs.values())) == 0:
            self.data_width = max_xy[0] - min_xy[0]
            self.data_height = max_xy[1] - min_xy[1]
            self.data_depth = max_xy[2] - min_xy[2]

            self.data_center = min_xy[0] + self.data_width/2, min_xy[1] + self.data_height/2, 0#min_xy[2] + self.data_depth/2
            # self.data_center = 0,0,0

        if zrange is not None:
             zmin, zmax = zrange
        else:
            zmin, zmax = min_xy[2], max_xy[2]
        # for mode in modes:
        program = PointShaderProgram(options=opts, zrange=(zmin, zmax), **kwargs)
        program.name = name
        program.setUniform('u_model', self.model)
        program.setUniform('u_view', self.view)
        program.setUniform('u_projection', self.projection)
        program.setAttributes(data)

        self.data_programs[name] = program

        return program

    def add_data_source_line(self, name, coords_start, coords_end, **args):
        #interleave coordinates
        min_xy = np.nanmin( coords_start, axis=0 )
        max_xy = np.nanmax( coords_start, axis=0 )
        if len(list(self.data_programs.values())) == 0:
            self.data_width = max_xy[0] - min_xy[0]
            self.data_height = max_xy[1] - min_xy[1]
            self.data_depth = max_xy[2] - min_xy[2]

            self.data_center = min_xy[0] + self.data_width/2, min_xy[1] + self.data_height/2, min_xy[2] + self.data_depth/2
        m,n = coords_start.shape
        vertices = np.empty((m*2,n), dtype=coords_start.dtype)
        vertices[0::2] = coords_start
        vertices[1::2] = coords_end
      
        data = np.empty( 2*m, [('a_position', np.float32, 3)] )
        data['a_position'] = vertices

        program = LineShaderProgram(**args)

        program.name = name
        program.setUniform('u_model', self.model)
        program.setUniform('u_view', self.view)
        program.setUniform('u_projection', self.projection)
        program.setAttributes(data)
        self.data_programs[name] = program

        return program

    def add_data_source_ball(self, name, points, radii, color=(0,1,0)):
        program = BallShaderProgram(points, radii, color)
        program.name = name
        program['u_model'] = self.model
        program['u_view'] = self.view
        program['u_projection'] = self.projection
        self.data_programs[name] = program

        return program

    def update_view_matrix(self):
        self.view = np.eye(4, dtype=np.float32)
        translate(self.view, self.translation[0], self.translation[1], self.translation[2] )
        scale(self.view, self.scale, self.scale, self.scale)
        self.view = self.view.dot( np.array(q.matrix(self.rotation), dtype=np.float32) )
        # translate(self.view, -self.translation[0], -self.translation[1], -self.translation[2] )
        translate(self.view, 0,0, self.camera_position)
        for program in list(self.data_programs.values()):
            program.setUniform('u_view', self.view)
            if program.draw_type == gl.GL_POINTS:
                program.setUniform('u_model_scale', self.scale)

    def update_projection_matrix(self):
        view_width, view_height = [x/self.radius for x in self.size]

        if self.projection_mode == 'orthographic':
            self.projection = ortho(-view_width, view_width, -view_height, view_height, self.near_clip, self.far_clip)
        elif self.projection_mode == 'perspective':
            self.projection = perspective(self.fov, view_width / float(view_height), self.near_clip, self.far_clip)

        for program in list(self.data_programs.values()):
            program.setUniform('u_projection', self.projection)

    def screen2view(self, x,y):
        width, height = self.size
        # print (x-width/2.)/self.radius, ((height-y)-height/2.)/self.radius
        return (x-width/2.)/self.radius, ((height-y)-height/2.)/self.radius

    def set_bg(self):
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        if self.bg_white:
            gl.glClearColor(0.95,0.95,0.95,1)
            # gloo.set_state('translucent', clear_color=np.array([1,1,1,1]) )
        else:
            gl.glClearColor(0.05,0.05,0.05,1)
            # gloo.set_state('translucent', clear_color=np.array([0.15,0.15,0.15,1]) )

    def keyPressEvent(self, event):
        key = event.key()
        repeat = event.isAutoRepeat()
        if key == Qt.Key_R:
            self.view = np.eye(4, dtype=np.float32)
            self.rotation = q.quaternion()
            self.scale = self.modelscale
            self.camera_position = -12.
            self.near_clip = 2.
            self.far_clip = 100.
            self.center_view()
            self.update_projection_matrix()
        elif key == Qt.Key_Minus:
            for program in list(self.data_programs.values()):
                if program.draw_type == gl.GL_POINTS:
                    if (program.is_visible and self.multiview) or not self.multiview:
                        program.setUniform('u_point_size', program.uniforms['u_point_size']/1.2)
        elif key == Qt.Key_Equal:
            for program in list(self.data_programs.values()):
                if program.draw_type == gl.GL_POINTS:
                    if (program.is_visible and self.multiview) or not self.multiview:
                        program.setUniform('u_point_size', program.uniforms['u_point_size']*1.2)
        elif key == Qt.Key_B:
            for program in list(self.data_programs.values()):
                if program.is_visible and program.draw_type == gl.GL_POINTS:
                    program.do_blending = not program.do_blending
        elif key == Qt.Key_C:
            self.near_clip = -self.camera_position - 0.1
            self.far_clip = -self.camera_position + 0.1
            self.update_projection_matrix()
        elif key == Qt.Key_T:
            self.rotation = q.quaternion()
            self.update_view_matrix()
        elif key == Qt.Key_P:
            if self.projection_mode == 'perspective':
                self.projection_mode = 'orthographic'
            else:
                self.projection_mode = 'perspective'
            self.update_projection_matrix()
        elif key == Qt.Key_L:
            self.bg_white = not self.bg_white
            self.set_bg()
        elif Qt.Key_0 <= key <= Qt.Key_9:
            i = int(chr(key))-1
            if i < len(list(self.data_programs.keys())):
                if self.multiview:
                    self.set_layer_visibility(list(self.data_programs.keys())[i], not list(self.data_programs.values())[i].is_visible)
                else:
                    for pi, prog in enumerate(self.data_programs.values()):
                        prog.is_visible = False
                        if pi == i:
                            prog.is_visible = True

        self.update()

    def on_resize(self, size_x, size_y):
        gl.glViewport(int(0), int(0), int(size_x), int(size_y))

        self.radius = 0.5 * min(size_x, size_y)
        self.size = size_x, size_y

        self.update_projection_matrix()

    def wheelEvent(self, event):
    # def on_mouse_wheel(self, window, offset_x, offset_y):
        ticks = float(event.angleDelta().y())/50
        modifiers = event.modifiers()

        if modifiers == Qt.ControlModifier | Qt.AltModifier:
            # if modifiers == Qt.ShiftModifier:
            ticks /= 30
            self.near_clip -= ticks
            self.far_clip -= ticks
            self.update_projection_matrix()
        elif modifiers == Qt.AltModifier:
            # if modifiers == Qt.ShiftModifier:
            ticks /= 30
            new = max(0.1,self.near_clip - ticks)
            if new <= self.far_clip:
                self.near_clip = new
                self.update_projection_matrix()
        elif modifiers == Qt.ControlModifier:
            # if modifiers == Qt.ShiftModifier:
            ticks /= 30
            new = min(1000,self.far_clip - ticks)
            if new >= self.near_clip:
                self.far_clip = new
                self.update_projection_matrix()
        elif modifiers == Qt.ShiftModifier:
            if self.projection_mode == 'perspective':
                old_fov = self.fov
                # do `dolly zooming` so that world appears at same size after canging fov
                self.fov = max(5.,self.fov + ticks)
                self.fov = min(120.,self.fov)
                self.camera_position = self.camera_position * (math.tan(math.radians(old_fov)/2.)) / (math.tan(math.radians(self.fov)/2.))
                self.update_projection_matrix()
                self.update_view_matrix()
        elif modifiers == Qt.MetaModifier:
            self.camera_position += ticks/10
            self.update_view_matrix()
        else:
            self.scale *= ticks/10 + 1.
            # self.camera_position += ticks/10
            self.update_view_matrix()
        self.update()

    def mouseMoveEvent(self, event):
        modifiers = event.modifiers()
        buttons = event.buttons()
        pos_x, pos_y = event.x(), event.y()

        if Qt.ShiftModifier == modifiers:
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
            
            self.hud_program.is_visible = True
        elif Qt.LeftButton == buttons:
            x0,y0 = self.screen2view(*self.last_mouse_pos)
            x1,y1 = self.screen2view(pos_x, pos_y)

            v0 = q.arcball(x0, y0)
            v1 = q.arcball(x1, y1)

            self.rotation = q.product(v1, v0, self.rotation)

            self.hud_program.is_visible = True
        else:
            self.hud_program.is_visible = False
        self.update_view_matrix()
        self.update()

        self.last_mouse_pos = pos_x, pos_y

    def update(self):
        self.render()


if __name__ == '__main__':

    import sys

    window = App()
    
    window.run()
