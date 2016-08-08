# some refs to implenting splatting:
# http://stackoverflow.com/questions/8608844/resizing-point-sprites-based-on-distance-from-the-camera
# https://github.com/AndreaMelle/pointBasedRendering/blob/master/resources/shaders/pointVertexShader.vert
# https://docs.google.com/document/d/1gbH_H2brkeE4bxU56GlrULLP8EfZV2160unzBNAsk04/edit?pli=1
# https://github.com/potree/potree/issues/37

# import geometry

import OpenGL.GL as gl
import ctypes
import numpy as np

class SimpleShaderProgram(object):
    def __init__(self, options=[], draw_type='points', **args):
        self.draw_type = draw_type
        if 'is_visible' in args:
            self.is_visible = args['is_visible']
        else:
            self.is_visible = False
        self.uniforms = {}
        self.color = (1,0,0)
        self.defines = ''

        if 'alternate_vcolor' in options:
            self.defines += '#define alternate_vcolor\n'

        self.default_mask = None
        if 'default_mask' in args:
            self.default_mask = args['default_mask']

        self.draw_types = {'points': gl.GL_POINTS, 'lines': gl.GL_LINES, 'triangles': gl.GL_TRIANGLES, 'line_strip':gl.GL_LINE_STRIP, 'line_loop':gl.GL_LINE_LOOP}

    def initialise(self):
        def compileShader(handle, shader_source):
            gl.glShaderSource(handle, shader_source)
            gl.glCompileShader(handle)
            if not gl.glGetShaderiv(handle, gl.GL_COMPILE_STATUS):
                print(gl.glGetShaderInfoLog(handle))

        self.program = gl.glCreateProgram()
        vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        
        compileShader(vertex_shader, self.vertex_str())
        compileShader(fragment_shader, self.fragment_str())

        gl.glAttachShader(self.program, vertex_shader)
        gl.glAttachShader(self.program, fragment_shader)
        gl.glLinkProgram(self.program)

        gl.glDetachShader(self.program, vertex_shader)
        gl.glDetachShader(self.program, fragment_shader)

        # Request a buffer slot from GPU
        self.buffer = gl.glGenBuffers(1)
        self.VAO = gl.glGenVertexArrays(1)

    def delete(self):
        # gl.glDeleteVertexArrays(1, self.VAO)
        # gl.glDeleteBuffers(1, self.buffer)
        gl.glDeleteProgram(self.program)

    def updateAttributes(self, filter=None):
        # set the correct filter/mask
        if filter is None:
            if self.default_mask is None:
                data = self.data
            else:
                data = self.data[self.default_mask]
        else:
            data = self.data[filter]
        self.dataLen = data.shape[0]
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def setAttributes(self, data):
        self.data = data
        self.dataLen = data.shape[0]
        # Make this buffer the default one
        gl.glBindVertexArray(self.VAO)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer)
        # Upload data
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

        offset = 0
        for i, name in enumerate(data.dtype.names):
            # import ipdb; ipdb.set_trace()
            stride = data[name].strides[0]
            if data[name].ndim == 1:
                size = 1
            else:
                size = data[name].shape[1]
            loc = gl.glGetAttribLocation(self.program, name)
            
            gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, False, stride, ctypes.c_void_p(offset))
            gl.glEnableVertexAttribArray(loc)
            offset += data.dtype[name].itemsize

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        # ensure correct mask is rendered:
        self.updateAttributes()

    def setUniform(self, name, data):
        gl.glUseProgram(self.program)
        loc = gl.glGetUniformLocation(self.program, name)
        if loc != -1:
            self.uniforms[name] = data
            if type(data) in [float, np.float64, np.float32]:
                gl.glUniform1f(loc, np.float32(data))
            elif type(data) == np.ndarray and data.shape == (4,4):
                gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, np.float32(data))
        gl.glUseProgram(0)

    def draw(self):
        if self.is_visible:
            # self.setAttributes(self.data)
            gl.glUseProgram(self.program)
            gl.glBindVertexArray(self.VAO)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer)
            gl.glDrawArrays(self.draw_types[self.draw_type], 0, self.dataLen)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)

    def toggle_visibility(self):
        self.is_visible = not self.is_visible
    
    def vertex_str(self):
        return """
        #version 330

        {defines}

        // Uniforms
        // ------------------------------------
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;

        // Attributes
        // ------------------------------------
        in vec3 a_position;

        out vec4 vcolor;

        void main (void) {{

            vcolor = vec4({color[0]}, {color[1]}, {color[2]}, 1);
            #if defined(alternate_vcolor)
            if (gl_VertexID%2==0) {{
                vcolor = vec4(1,1,1, 1);
            }}
            #endif
            
            gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);    
        }}
        """.format(color=self.color, defines=self.defines)

    def fragment_str(self):
        return """
        #version 330

        in vec4 vcolor;
        out vec4 color;

        void main()
        {{
            color =  vcolor;
        }}
        """

class CrossHairProgram(SimpleShaderProgram):
    hud_data = np.zeros( 4, [('a_position', np.float32, 2)] )
    hud_data['a_position'] = np.array([[-1, 0], [1,0], [0,-1], [0,1]], dtype=np.float32)

    def __init__(self, **args):
        super(CrossHairProgram, self).__init__(draw_type='lines', **args)
        self.initialise()
        self.setAttributes(self.hud_data)

    def vertex_str(self):
        return """
        #version 330

        in vec2  a_position;

        void main (void) {
            gl_Position =  vec4(a_position, 0.0, 1.0);
        }
        """

    def fragment_str(self):
        return """
        #version 330

        out vec4 color;

        void main()
        {
            color = vec4(0.5,0.5,0.5,1.0);
        }
        """

class PointShaderProgram(SimpleShaderProgram):
    _all_modes = ['with_normals', 'with_point_radius', 'with_intensity', 'splat_disk', 'splat_point', 'adaptive_point', 'fixed_point', 'fixed_color']
    
    def __init__(self, options=['fixed_point'], **kwargs):
        super(PointShaderProgram, self).__init__(options, draw_type='points', **kwargs)

        self.zmin, self.zmax = kwargs['zrange']

        options += ['with_texture']
        for option in options:
            if option in self._all_modes:
                self.defines += "#define {}\n".format(option)
        # import ipdb; ipdb.set_trace()
        self.attributes = ""
        if 'with_normals' in options:
            self.attributes += "in vec3 a_normal;\n"
        if 'with_point_radius' in options:
            self.attributes += "in float a_splat_radius;\n"
        if 'with_intensity' in options:
            self.attributes += "in float a_intensity;\n"
        # import ipdb; ipdb.set_trace()

       
        if 'color' in kwargs:
            self.color = kwargs['color']
        self.initialise()

        self.setUniform('u_point_size', 20.0)
        if 'with_point_radius' in options:
            self.setUniform('u_point_size', 300.0)

        self.do_blending = False
        if 'blend' in options:
            self.do_blending = True

        colormap = 'jet'
        if 'colormap' in kwargs:
            colormap = kwargs['colormap']
        self.texture = self.create_colormap(scheme=colormap)
        

    def create_colormap(self, scheme='jet'):
        texture = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_1D, texture);
        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT);

        gl.glTexParameterf(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR);
        
        # Load and generate the texture
        width = 256
        if scheme == 'random':
            image = np.random.rand(width,3).astype(np.float32)
            image[0,:] = 1.
            # gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT);
            # gl.glTexParameterf(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            # gl.glTexParameterf(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST);
        else:
            from .colormaps import cm
            image = np.array(cm[scheme], dtype=np.float32)

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1);
        gl.glTexImage1D(gl.GL_TEXTURE_1D, 0, gl.GL_RGB32F, width, 0, gl.GL_RGB, gl.GL_FLOAT, image);
        gl.glBindTexture(gl.GL_TEXTURE_1D, 0);

        return texture

    def draw(self):
        if self.is_visible:
            # self.setAttributes(self.data)
            gl.glUseProgram(self.program)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_1D, self.texture);
            gl.glBindVertexArray(self.VAO)
            gl.glDrawArrays(self.draw_types[self.draw_type], 0, self.dataLen)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)

    def vertex_str(self):
        return """
        #version 330

        {defines}

        // Uniforms
        // ------------------------------------
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        uniform float u_point_size;

        #if defined(splat_point) | defined(splat_disk) | defined(adaptive_point)
            uniform float u_model_scale;
        #endif

        // Attributes
        // ------------------------------------
        in vec3  a_position;
        {attributes}

        // Varyings
        // ------------------------------------
        out float v_color_intensity;
        out float v_splat_radius;
        out float v_lightpwr;
        #if defined(with_normals)
            out vec4 v_normal;
        #endif

        void main (void) {{
            vec4 posEye = u_view * u_model * vec4(a_position, 1.0);    
            gl_Position = u_projection * posEye;

            #if defined(with_point_radius)
                float scale = u_model_scale*a_splat_radius;
            #elif defined(splat_point) | defined(splat_disk) | defined(adaptive_point)
                float scale = u_model_scale;
            #endif
            
            #if defined(splat_point) | defined(splat_disk) | defined(adaptive_point)
                vec4 projCorner = u_projection * vec4(scale, scale, posEye.z, posEye.w);
                float s = 2.0 * u_point_size / projCorner.w;
                gl_PointSize = projCorner.x * s;
            #else
                gl_PointSize = u_point_size;
            #endif
            //gl_PointSize = 0.0;
            
            #if defined(with_intensity)
                v_color_intensity = a_intensity;
            #else 
                v_color_intensity = (a_position[2]-({zmin}))/{zrange};
            #endif

            #if defined(with_normals)
                v_normal = u_view * u_model * vec4(a_normal, 0);
                v_normal = normalize(v_normal);

                vec4 lighting_direction_1 = vec4(1,1,1,0);
                vec4 lighting_direction_2 = vec4(0,0.5,1,0);
                float L = dot(v_normal, normalize(lighting_direction_1)) + 0.5*dot(v_normal, normalize(lighting_direction_2));
                v_lightpwr = clamp(abs(L), 0.3, 1);
            #else
                v_lightpwr = 1.0;
            #endif
        }}
        """.format(zrange=self.zmax-self.zmin, zmin=self.zmin, defines=self.defines, attributes=self.attributes)

    def fragment_str(self):
        return """
        #version 330

        {defines}


        uniform sampler1D u_color_ramp;


        // Constants
        // ------------------------------------


        // Varyings
        // ------------------------------------
        in float v_color_intensity;
        in float v_lightpwr;
        #if defined(with_normals)
            in vec4 v_normal;
        #endif

        out vec4 color;

        // Main
        // ------------------------------------
        void main()
        {{

            float x = gl_PointCoord.x - 0.5;
            float y = gl_PointCoord.y - 0.5;

            
            #if defined(splat_disk)
                float dz = -(v_normal.x/v_normal.z)*x + (v_normal.y/v_normal.z)*y;
            #elif defined(splat_point)
                float dz = 0;
            #endif

            float alpha = 1.;
            #if defined(splat_point) | defined(splat_disk)
                float r = length(vec3(x,y,dz));
                if (r > 0.5) {{
                    discard;
                }}
            #endif

            float c = 1.0 - (pow(2*x,2.0) + pow(2*x,2.0));
            //color =  vec4(color_scheme(v_color_intensity), alpha);
            //#else if defined(with_texture)
            #if defined(fixed_color)
                color =  v_lightpwr*vec4({color[0]}, {color[1]}, {color[2]}, 1);
            #else
                color =  v_lightpwr*texture(u_color_ramp, v_color_intensity);
            #endif
            gl_FragDepth = gl_FragCoord.z + 0.002*(1.0-pow(c, 1.0)) * gl_FragCoord.w;
            
        }}
        """.format(defines=self.defines, color=self.color)

### line shader:

class LineShaderProgram(SimpleShaderProgram):

    def __init__(self, **args):
        self.do_blending = False
        # self.zmin, self.zmax = zrange

        # self.defines = ""
        # for option in options:
        #     if option in all_options:
        #         self.defines += "#define {}\n".format(option)

        # if 'with_intensity' in options:
        #     self.attributes += "attribute float a_intensity;\n"

        super(LineShaderProgram, self).__init__(draw_type = 'lines', **args)
        if 'color' in args:
            self.color = args['color']
        self.initialise()

class TriangleShaderProgram(SimpleShaderProgram):

    def __init__(self, draw_type='triangles', **args):
        self.do_blending = False
        # self.zmin, self.zmax = zrange

        # self.defines = ""
        # for option in options:
        #     if option in all_options:
        #         self.defines += "#define {}\n".format(option)

        # if 'with_intensity' in options:
        #     self.attributes += "attribute float a_intensity;\n"
        super(TriangleShaderProgram, self).__init__(draw_type = draw_type, **args)
        if 'color' in args:
            self.color = args['color']
        self.initialise()

    def vertex_str(self):
        return """
#version 330

// Uniforms
// ------------------------------------
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

// Attributes
// ------------------------------------
in vec3  a_position;
in vec3  a_normal;

out vec4 v_color;

void main (void) {{
vec4 posEye = u_view * u_model * vec4(a_position, 1.0);    
gl_Position = u_projection * posEye;

vec4 n = u_view * u_model * vec4(a_normal, 0);
vec4 lighting_direction = vec4(1,1,1,0);
v_color = clamp(abs(dot(normalize(n), normalize(lighting_direction))), 0.3, 1) * vec4({0}, {1}, {2}, 1);
}}
""".format(self.color[0], self.color[1], self.color[2])

    def fragment_str(self):
        return """
#version 330

in vec4 v_color;
out vec4 color;

void main()
{{
color =  v_color;    
}}
"""


### triangle shader:

# class BallShaderProgram(gloo.Program):

#     def __init__(self, coords, radii, color):
#         self.draw_type = 'triangles'
#         self.is_visible = False
#         self.do_blending = False

#         self.color = color

#         m,n = coords.shape

#         resolution = 10
#         spheres = []
#         for i in xrange(m):
#             p, r = coords[i], radii[i]
#             sphere = geometry.create_sphere(cols=resolution, rows=resolution, radius=r)
#             sphere.set_vertices(verts=sphere.get_vertices()+p)
#             spheres.append(sphere)

#         lvert = spheres[0].get_vertices().shape[0]
#         lface = spheres[0].get_faces().shape[0]
#         normals_0 = spheres[0].get_vertex_normals()

#         vertices = np.empty((lvert*m, 3), dtype=np.float32)
#         normals = np.empty((lvert*m, 3), dtype=np.float32)
#         faces = np.empty((lface*m, 3), dtype=np.uint32)

#         for i, sphere in enumerate(spheres):
#             # import ipdb; ipdb.set_trace()
#             vertices[i*lvert:(i+1)*lvert,:] = sphere.get_vertices()
#             normals[i*lvert:(i+1)*lvert,:] = normals_0
#             faces[i*lface:(i+1)*lface,:] = sphere.get_faces() + i*lvert

#         data = np.empty( lvert*m, [('a_position', np.float32, 3), ('a_normal', np.float32, 3)] )
#         data['a_position'] = vertices
#         data['a_normal'] = normals
#         vbo = gloo.VertexBuffer(data)

#         self.indexbuffer = gloo.IndexBuffer(faces.reshape(-1))
        
#         # self.zmin, self.zmax = zrange

#         # self.defines = ""
#         # for option in options:
#         #     if option in all_modes:
#         #         self.defines += "#define {}\n".format(option)

#         # self.attributes = ""
#         # if 'with_normals' in options:
#         #     self.attributes += "attribute vec3 a_normal;\n"
#         # if 'with_point_radius' in options:
#         #     self.attributes += "attribute float a_splat_radius;\n"
#         # if 'with_intensity' in options:
#         #     self.attributes += "attribute float a_intensity;\n"

#         super(BallShaderProgram, self).__init__(self.vertex_str(), self.fragment_str())
#         self.bind(vbo)

#         # self['u_point_size'] = 10
#         # if 'with_point_radius' in options:
#         #     self['u_point_size'] = 300

#         # self.do_blending = False
#         # if 'blend' in options:
#         #     self.do_blending = True

#         # print self._user_variables

#     def toggle_visibility(self):
#         self.is_visible = not self.is_visible
    
#     def vertex_str(self):
#         return """
# #version 120

# // Uniforms
# // ------------------------------------
# uniform mat4 u_model;
# uniform mat4 u_view;
# uniform mat4 u_projection;

# // Attributes
# // ------------------------------------
# attribute vec3  a_position;
# attribute vec3  a_normal;

# varying vec4 v_color;

# void main (void) {{
#     vec4 posEye = u_view * u_model * vec4(a_position, 1.0);    
#     gl_Position = u_projection * posEye;

#     vec4 n = u_view * u_model * vec4(a_normal, 0);
#     vec4 lighting_direction = vec4(0,1,1,0);
#     v_color = clamp(abs(dot(normalize(n), normalize(lighting_direction))), 0, 1) * vec4({0}, {1}, {2}, 1);
# }}
# """.format(self.color[0], self.color[1], self.color[2])

#     def fragment_str(self):
#         return """
# varying vec4 v_color;

# void main()
# {{
#     gl_FragColor =  v_color;    
# }}
# """

# ## point shader:

