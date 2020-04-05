#SPECIAL THANKS TO IAN WILLIAMS FOR HELPING ME WITH SOME DEBUGGING, PRAISE PROGRAMMER JESUS, PEACE BE UPON HIM
#<3 IW
from math import cos, sin, radians, pi as PI
import os

# Core Matrix functionality
def matrix_mult( m1, m2 ):
    """ Multiply matrix m2 by m1, changing m2. """
    temp = new_matrix(len(m1), len(m2[0]))
    for row in range(len(temp)):
        for col in range(len(temp[0])):
            for i in range(len(m1[0])):
                temp[row][col] += m1[row][i] * m2[i][col]
    i = 0
    while i < len(m2):
        if m2[i]:
            m2[i] = temp[i]
        elif temp[i]:
            m2.append(temp[i])
        i += 1
    return m2
def display_matrix(x):
    for r in x:
        print(", ".join([str(round(i, 2)).ljust(5) for i in r]))

def ident( matrix ):
    for r in range( len( matrix[0] ) ):
        for c in range( len(matrix) ):
            if r == c:
                matrix[c][r] = 1
            else:
                matrix[c][r] = 0

def new_matrix(rows = 4, cols = 4):
    m = []
    for r in range(rows):
        m.append([])
        for c in range(cols):
            m[r].append(0)
    return m
    # IW The bottom code seems to be reversed. It should be like above
    for c in range( cols ):
        m.append( [] )
        for r in range( rows ):
            m[c].append( 0 )
    return m


# Misc. useful matrixes to have
def make_hermite():
    t = [[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]]
    return t

def make_translate( x, y, z ):
    t = new_matrix()
    ident(t)
    t[0][3] = x
    t[1][3] = y
    t[2][3] = z
    return t

def make_scale( x, y, z ):
    t = new_matrix()
    ident(t)
    t[0][0] = x
    t[1][1] = y
    t[2][2] = z
    return t

def make_rotX( theta ):
    t = new_matrix()
    ident(t)
    t[1][1] = cos(theta)
    t[1][2] = - sin(theta)
    t[2][1] = sin(theta)
    t[2][2] = cos(theta)
    return t

def make_rotY( theta ):
    t = new_matrix()
    ident(t)
    t[0][0] = cos(theta)
    t[0][2] = sin(theta)
    t[2][0] = - sin(theta)
    t[2][2] = cos(theta)
    return t

def make_rotZ( theta ):
    t = new_matrix()
    ident(t)
    t[0][0] = cos(theta)
    t[0][1] = - sin(theta)
    t[1][0] = sin(theta)
    t[1][1] = cos(theta)
    return t


def circle_point(x, y, theta, r):
    """ Returns a point in a polar way. Centered at point (x, y), returns the point r away at an angle of theta degrees. """
    xn = x + (r * cos(radians(theta)))
    yn = y + (r * sin(radians(theta)))
    return [xn, yn]


# Vector related function
def dot_product(a, b):
    """ Return the dot product of 2 vectors. """
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])
def normal(m, i):
    """ Returns the normal of a polygon in a given matrix. """
    a = [m[0][i + 1] - m[0][i],
	 m[1][i + 1] - m[1][i],
	 m[2][i + 1] - m[2][i]]
    b = [m[0][i + 2] - m[0][i],
	 m[1][i + 2] - m[1][i],
	 m[2][i + 2] - m[2][i]]
    n = [(a[1] * b[2]) - (a[2] * b[1]),
	 (a[2] * b[0]) - (a[0] * b[2]),
	 (a[0] * b[1]) - (a[1] * b[0])];
    return n

# Point generation for useful 3d shapes. And toruses...
## Useful globals
### 3d shape resolution
S_RES = 30 # Resolution for sphere
T_RES = 20 # Resolution for torus

def generate_sphere(x, y, z, r):
    """ Generates the points on a sphere, stores them in a 3xn array, and returns them. """
    m = new_matrix(3, 0)
    for t in range(2 * S_RES):
        theta = t * PI / S_RES
        phi = 0
        for p in range(S_RES):
            phi = p * PI / (S_RES - 1)
            m[0].append(x + (r * cos(theta) * sin(phi)))
            m[1].append(y + (r * sin(theta) * sin(phi)))
            m[2].append(z + (r * cos(phi)))
    return m

def generate_torus(x, y, z, r, R):
    """ Generates the points on a torus, stores them in a 3xn array, and returns them. """
    m = new_matrix(3, 0)
    for t in range(2 * T_RES):
        theta = t * PI / T_RES
        for p in range(2 * T_RES):
            phi = p * PI / T_RES
            m[0].append(x + (cos(theta) * (r * cos(phi) + R)))
            m[1].append(y + (r * sin(phi)))
            m[2].append(z - (sin(theta) * (r * cos(phi) + R)))
    return m


class Picture:
    """ A class that represents a canvas, upon which artists can create beautiful works."""
    def __init__(self, n, w, h):
        self.name = n
        self.view_vector = [0, 0, 1]
        self.width, self.height = (w, h)
        self.pixels = [[[0, 0, 0] for i in range(self.width)] for j in range(self.height)]
        self.four_identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.edge_matrix = [[], [], [], []]
        self.triangle_matrix = [[], [], [], []]
        self.transformation_matrix = ident(new_matrix())

    def plot(self, x, y, color):
        """ Plot a point on the screen. """
        if x < 0 or y < 0 or x >= self.width or y >= self.height:#out of bounds
            return
        self.pixels[int(self.height - y - 1)][int(x)] = color #Flip for humans

    def draw_line(self, x0, y0, x1, y1, color ):
        """ Draw a line on the screen. """
        #swap points if going right -> left
        if x0 > x1:
            xt = x0
            yt = y0
            x0 = x1
            y0 = y1
            x1 = xt
            y1 = yt

        x = x0
        y = y0
        A = 2 * (y1 - y0)
        B = -2 * (x1 - x0)

        #octants 1 and 8
        if ( abs(x1-x0) >= abs(y1 - y0) ):

            #octant 1
            if A > 0:
                d = A + B/2

                while x < x1:
                    self.plot(x, y, color)

                    if d > 0:
                        y+= 1
                        d+= B
                    x+= 1
                    d+= A
                #end octant 1 while
                self.plot(x1, y1, color)
            #end octant 1

            #octant 8
            else:
                d = A - B/2

                while x < x1:
                    self.plot(x, y, color)
                    if d < 0:
                        y-= 1
                        d-= B
                    x+= 1
                    d+= A
                #end octant 8 while
                self.plot(x1, y1, color)
            #end octant 8
        #end octants 1 and 8

        #octants 2 and 7
        else:
            #octant 2
            if A > 0:
                d = A/2 + B

                while y < y1:
                    self.plot(x, y, color)
                    if d < 0:
                        x+= 1
                        d+= A
                    y+= 1
                    d+= B
                #end octant 2 while
                self.plot(x1, y1, color)
            #end octant 2

            #octant 7
            else:
                d = A/2 - B;

                while y > y1:
                    self.plot(x, y, color)
                    if d > 0:
                        x+= 1
                        d+= A
                    y-= 1
                    d-= B
                #end octant 7 while
                self.plot(x1, y1, color)

    def display_edge_matrix(self):
        for i in self.edge_matrix:
            print(i)
    def add_3d_point(self, x, y, z, m=None):
        """ Add a single point to the edge matrix by default or triangle matrix if specified. """
        if not m:
            m = self.edge_matrix
        m[0].append(x)
        m[1].append(y)
        m[2].append(z)
        m[3].append(1)
    def add_edge(self, x0, y0, z0, x1, y1, z1):
        """ Add an edge to the edge matrix. """
        self.add_3d_point(x0,y0,z0)
        self.add_3d_point(x1,y1,z1)
    def add_polygon(self, x0, y0, z0, x1, y1, z1, x2, y2, z2):
        """ Add a polygon to the edge matrix. """
        self.add_3d_point(x0,y0,z0, m=self.triangle_matrix)
        self.add_3d_point(x1,y1,z1, m=self.triangle_matrix)
        self.add_3d_point(x2,y2,z2, m=self.triangle_matrix)
    def add_circle(self, cx, cy, cz, r, step ):
        """ Add a circle to the edge matrix. """
        centx = cx
        centy = cy
        theta = 0
        last_pos = circle_point(centx, centy, 0, r)
        theta += step
        while(theta <= 360 + step):
            secondpoints = circle_point(centx, centy, theta, r)
            self.add_edge(last_pos[0], last_pos[1], 1, secondpoints[0], secondpoints[1], 1)
            #centx = secondpoints[0]
            #centy = secondpoints[1]
            last_pos = secondpoints
            theta += step

    def add_curve(self, x0, y0, x1, y1, x2, y2, x3, y3, step, curve_type ):
        """ Add a curve, either hermite or bezier, to the edge matrix. """
        t = 0
        if curve_type == "bezier":
            while t <= 1:
                d = [pow(1 - t, 3), 3 * t * pow(1 - t, 2), 3 * pow(t, 2) * (1 - t), pow(t, 3)]
                x = (x0 * d[0]) + (x1 * d[1]) + (x2 * d[2]) + (x3 * d[3])
                y = (y0 * d[0]) + (y1 * d[1]) + (y2 * d[2]) + (y3 * d[3])
                self.add_3d_point(x, y, 0)
                t += step
        else:
            m = make_hermite()
            pmatrix = [[x0, y0, 1], [x1, y1, 1], [x2, y2, 1], [x3, y3, 1]]
            while t <= 1:
                tmatrix = [[t * t * t, t * t, t, 1]]
                a = matrix_mult(matrix_mult(tmatrix, m), pmatrix)
                self.add_3d_point(a[0][0], a[0][1], a[0][2])
                t += step

    def add_box(self, x, y, z, width, height, depth):
        """
        Add a box to the edge matrix. Rectangular Prism is such an ugly word, using box means instead means I take up less space.

        points (for reference):
          4_____7
         /:    /|
        0_____3 |
        | 5---|-6
        |/    |/
        1_____2
        """
        points = [[x, y, z], [x, y - height, z], [x + width, y - height, z], [x + width, y, z],
		  [x, y, z - depth], [x, y - height, z - depth], [x + width, y - height, z - depth], [x + width, y, z - depth]]
        # add_polygon returns null, so the lambda makes a rectangular face by returning an or statement of the add_polygon calls on the 2 triangles that make it up. Simplifies code.
        add_side = lambda screen, a, b, c, d : (
            screen.add_polygon(points[a][0], points[a][1], points[a][2],
                               points[b][0], points[b][1], points[b][2],
                               points[c][0], points[c][1], points[c][2])
            or
            screen.add_polygon(points[c][0], points[c][1], points[c][2],
                               points[d][0], points[d][1], points[d][2],
                               points[a][0], points[a][1], points[a][2]))
        add_side(self, 0, 1, 2, 3);
        add_side(self, 4, 0, 3, 7);
        add_side(self, 3, 2, 6, 7);
        add_side(self, 2, 1, 5, 6);
        add_side(self, 4, 5, 1, 0);
        add_side(self, 7, 6, 5, 4);
    def add_sphere(self, x, y, z, r):
        """
        Add a sphere to the polygon matrix.
           a---i      |  a - i            |   a   i
           A          |  | X |            |     V
         q - t        |  q - t            | q---t
        [i, res - 1)  |  [i + 1, res - 2) | [res - 1]

        res = 5:
            10---5/0          TQA
          11    6   1    ITA, TQA
        12     7     2   ITA, TQA
          13    8   3    ITA,
            14---9/4

        """



        m = generate_sphere(x, y, z, r)
        for i in range(len(m[0])):
            a = (i + 1)
            t = (i + S_RES) % (2 * S_RES * S_RES)
            q = (a + S_RES) % (2 * S_RES * S_RES)
            if (i + 1) % S_RES ==  0:
                continue
            if i % S_RES > 0:
                self.add_polygon(m[0][i], m[1][i], m[2][i],
                                 m[0][t], m[1][t], m[2][t],
                                 m[0][a], m[1][a], m[2][a])
            if (i + 2) % S_RES > 0:
                self.add_polygon(m[0][t], m[1][t], m[2][t],
                                 m[0][q], m[1][q], m[2][q],
                                 m[0][a], m[1][a], m[2][a])
    def add_torus(self, x, y, z, r, R):
        """
        Add a torus to the edge matrix. R is the radius of the torus, r is the radius of the circle.

          a - i
          | X |
          q - t


        res = 4:
           8---------0
         15_9-------7_1
        14___10----6___2
         13_11------5_3
          12---------4
        """

        m = generate_torus(x, y, z, r, R)
        for i in range(len(m[0])):
             a = i + 1
             a = a if (a % (2 * T_RES)) else a - 2 * T_RES;
             t = (i + 2 * T_RES) % (4 * T_RES * T_RES)
             q = (a + 2 * T_RES) % (4 * T_RES * T_RES)
             self.add_polygon(m[0][i], m[1][i], m[2][i],
                              m[0][t], m[1][t], m[2][t],
                              m[0][a], m[1][a], m[2][a])

             self.add_polygon(m[0][t], m[1][t], m[2][t],
                              m[0][q], m[1][q], m[2][q],
                              m[0][a], m[1][a], m[2][a])
    def draw_lines(self):
        """ Draw the edges in the edge matrix. """
        matrix = self.edge_matrix
        if len(matrix) < 2:
            print('Need at least 2 points to draw')
            return
        point = 0
        while point < len(matrix[0]) - 1:
            self.draw_line( int(matrix[0][point]),
                int(matrix[1][point]),
                int(matrix[0][point+1]),
                int(matrix[1][point+1]),
                       [255, 255, 255])
            point+= 2

    def draw_polygons(self):
        matrix = self.triangle_matrix
        if len(matrix) < 3:
            print('Need at least 3 points to draw')
            return
        point = 0
        while point < len(matrix[0]) - 1:
            if dot_product(normal(matrix, point), self.view_vector) >= 0:
                self.draw_line( int(matrix[0][point]), int(matrix[1][point]),
                                int(matrix[0][point+1]),int(matrix[1][point+1]),
                                [255, 255, 255])
                self.draw_line( int(matrix[0][point]), int(matrix[1][point]),
                                int(matrix[0][point+2]),int(matrix[1][point+2]),
                                [255, 255, 255])
                self.draw_line( int(matrix[0][point+2]), int(matrix[1][point+2]),
                                int(matrix[0][point+1]),int(matrix[1][point+1]),
                                [255, 255, 255])
            point += 3
    def clear_screen(self):
        """ Clear the pixels in the screen. """
        self.pixels = [[[0, 0, 0] for i in range(self.width)] for j in range(self.height)]
    def clear_edge_matrix(self):
        """ Clear the edge matrix. Useful for removing shapes you no longer want anything to do with. """
        self.edge_matrix = [[], [], [], []]
    def clear_triangle_matrix(self):
        """ Clear the triangle matrix. Useful for removing shapes you no longer want anything to do with. """
        self.triangle_matrix = [[], [], [], []]        
    def pixels_to_ascii(self):
        """ Returns the picture in ppm string format. """
        s = ""
        for x in range(self.width):
            for y in range(self.height):
                n = self.pixels[x][y]
                s += str(n[0]) + " " + str(n[1]) + " " + str(n[2]) + "  "
                s += "\n"
        return s

    def save(self):
        """ Writes the picture to a file. """
        n = str(self.name)
        f = open(n + ".ppm", "w")
        f.write("P3\n" + str(self.width) + " " + str(self.height) + "\n255\n")
        f.write(self.pixels_to_ascii())
        f.close()
        print(n + '.ppm')

# Commands in a script file for which arguments must be given
ARG_COMMANDS = [ 'line', 'scale', 'move', 'rotate', 'save', 'bezier', 'hermite', 'circle', 'box', 'sphere', 'torus']

def parse_file( fname, transform, screen, color ):
    """ Parse the script and carry out the appropriate commands. """
    f = open(fname)
    lines = f.readlines()
    c = 0
    screen.clear_screen()
    stack = [transform]
    while c < len(lines):
        #print "--------------------------------"
        #display_matrix(stack[-1])
        
        line = lines[c].strip()
        if line in ARG_COMMANDS:
            c += 1
            args = lines[c].strip().split(' ')
        if line =='push':
            stack.append([[col for col in row] for row in stack[-1]])
        elif line == 'pop':
            stack.pop()
        elif line == 'line':
            screen.add_edge(
                      float(args[0]), float(args[1]), float(args[2]),
                      float(args[3]), float(args[4]), float(args[5]) )
            matrix_mult(stack[-1], screen.edge_matrix )
            screen.clear_edge_matrix()
            screen.draw_lines()
        elif line == 'circle':
            screen.add_circle(float(args[0]), float(args[1]), float(args[2]), float(args[3]), 1)
            matrix_mult(stack[-1], screen.edge_matrix )
            screen.draw_lines()
            screen.clear_edge_matrix()
        elif line == 'bezier':
            screen.add_curve(float(args[0]), float(args[1]), float(args[2]),
                      float(args[3]), float(args[4]), float(args[5]),
                      float(args[6]), float(args[7]), 0.005, "bezier")
            matrix_mult(stack[-1], screen.edge_matrix )
            screen.draw_lines()
            screen.clear_edge_matrix()
            
        elif line == 'hermite':
            screen.add_curve(float(args[0]), float(args[1]), float(args[2]),
                      float(args[3]), float(args[4]), float(args[5]),
                      float(args[6]), float(args[7]), 0.005, "hermite")
            matrix_mult(stack[-1], screen.edge_matrix )
            screen.draw_lines()
            screen.clear_edge_matrix()
            
        elif line == 'box':
            screen.add_box(
                float(args[0]), float(args[1]), float(args[2]),
                float(args[3]), float(args[4]), float(args[5]) )
            matrix_mult(stack[-1], screen.triangle_matrix )
            screen.draw_polygons()
            screen.clear_triangle_matrix()
            
        elif line == 'sphere':
            screen.add_sphere(float(args[0]), float(args[1]), float(args[2]), float(args[3]))
            matrix_mult(stack[-1], screen.triangle_matrix )
            screen.draw_polygons()
            screen.clear_triangle_matrix()
        elif line == 'torus':
            screen.add_torus(float(args[0]), float(args[1]), float(args[2]), float(args[3]), float(args[4]))
            matrix_mult(stack[-1], screen.triangle_matrix )
            screen.draw_polygons()
            screen.clear_triangle_matrix()
        elif line == 'scale':
            t = make_scale(float(args[0]), float(args[1]), float(args[2]))
            matrix_mult(stack[-1], t)
            stack[-1] = t
        elif line == 'move':
            t = make_translate(float(args[0]), float(args[1]), float(args[2]))
            matrix_mult(stack[-1], t)
            stack[-1] = t
            
        elif line == 'rotate':
            theta = float(args[1]) * (PI / 180)
            if args[0] == 'x':
                t = make_rotX(theta)
            elif args[0] == 'y':
                t = make_rotY(theta)
            else:
                t = make_rotZ(theta)
            matrix_mult(stack[-1], t)
            stack[-1] = t
        elif line == 'display' or line == 'save':
            screen.save()
            if line == 'display':
                os.system('display *.ppm')
            else:
                pass

        c+= 1

def main():
    screen = Picture('image', 500, 500)
    transform = [ [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    color = [255, 255, 255]
    ident(transform)
    parse_file( 'script', transform, screen, color )

if __name__ == "__main__":
    main()
