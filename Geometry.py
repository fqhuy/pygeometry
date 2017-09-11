# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 09:14:28 2014

@author: phan
"""
from numpy.linalg import norm
import math
import numpy as np
from numpy import dot
from scipy.spatial.distance import euclidean as euc
from shapely.geometry import *
from skimage import draw
try:
    from freetype import *
except Exception as e:
    print('need freetype-py for font-related functionality')

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
#from Common import eps
from . import geometry as geo
import networkx as nx
#infinity point
inf_point = Point(np.finfo('f').min,np.finfo('f').min)
infinity = np.finfo('d').max

eps = 1e-9


def atan3(y, x):
    """
    360 degrees atan
    """
    a = np.arctan2(y, x)
    if not hasattr(a, '__len__'):
        if a < 0:
            return np.pi * 2 + a
        else:
            return a
        
    for i in range(len(a)):
        if a[i] < 0:
            a[i] += np.pi * 2
            
    return a


def adiff(u, v):
    diff = []
    for i in range(len(u)):
        if u[i] > 180:
            u[i] -= 360
        
        if v[i] > 180:
            v[i] -= 360
            
        diff.append(abs(u[i]-v[i]))
        
    return np.array(diff)


def make_mask(polygons, shape, offset=[0,0], scale=1):
    offset = np.array(offset)
    w, h = shape #image.shape[:2]
    masks = []
    for poly in polygons:
        if len(poly) < 3:
            continue
        
        mask = np.zeros((h, w), dtype='uint8')
        hl = np.array(poly) * scale + offset
        rr, cc = draw.polygon(hl[:,1], hl[:,0], mask.shape)
        mask[rr, cc] = 1
        masks.append(mask)
    
    #holemask = np.ones(len(masks), dtype='uint8')
    #take care of the holes
    for mx in range(len(masks)):
        for my in range(len(masks)):
            if (masks[my] == 0).all() or (masks[mx] == 0).all():
                continue
            
            if mx!=my and (np.bitwise_and(masks[mx], masks[my]) == masks[my]).all() :
                #holemask[my] = 0
                masks[mx] = np.bitwise_and(masks[mx], np.bitwise_not( masks[my] ))
                masks[my] *= 0 

    #masks = [m for m in masks if (m != 0).sum() > 400]
    #return masks
    combined_mask = sum(masks)
    combined_mask[combined_mask != 0] = 1
    return combined_mask
 
def make_masks(anns, attrs, image, offset=[0,0], scale = 1):
    masks = {}
    
    #for ax, ann in enumerate(anns):
    #    masks.append(make_mask(ann, image, offset, scale) )
    for gx, group in anns.iteritems():
        masks[gx] = make_mask(group['polies'][0], image.shape[:2], offset, scale)
    
    for mx in masks.keys():
        for my in masks.keys():
            if mx!=my and \
            (np.bitwise_and(masks[mx], masks[my]) == masks[my]).all() and \
            attrs[my][4]==1:
                masks[mx] = np.bitwise_and(masks[mx], np.bitwise_not( masks[my]) )
                
        #masks[mx] *= (mx + 1)
    
    return masks


def circle_to_poly(center, r, astep=10):
    i = 0
    points = []
    while i < np.pi * 2:
        points.append([np.math.cos(i) * r +  center[0], np.math.sin(i) * r + center[1]])
        i += np.math.radians(astep)
        
    return points
    
def bounding_box(paths):
    '''
    find the bounding box of a list of paths
    '''
    pass

def line_intersection(p1, p2, p3, p4):
    x43 = p4[0] - p3[0]
    y43 = p4[1] - p3[1]
    x31 = p3[0] - p1[0]
    y31 = p3[1] - p1[1]
    x21 = p2[0] - p1[0]
    y21 = p2[1] - p1[1]
    s = (x43 * y31 - x31 * y43) / (x43 * y21 - x21 * y43)
    t = (x21 * y31 - x31 * y21) / (x43 * y21 - x21 * y43)
    intersection = None
    if s >= 0 and s <= 1 and t >=0 and t <= 1:
        intersection = np.array([p1[0] + (p2[0] - p1[0]) * s, p1[1] + (p2[1] - p1[1]) * s])
    
    return intersection

def rotate_matrix(vec1, vec2):
    cos = np.round(vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), decimals=5)
    sin = np.cross( vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2))
    rot_matrix = np.array([[cos, -sin],[sin, cos]])
    #nvec2 = rot_matrix.dot( nvec1 )
    return rot_matrix


def sample_polyline(poly, step=None, n_samples=None):
    poly_ = asLineString(poly)
    re = []

    if poly_.length == 0.0:
        return np.tile(poly[0], n_samples).reshape((n_samples, -1))

    if step is None and n_samples is not None:
        step = poly_.length / n_samples
        
    # if step < 1:
    #     bound = 1
    # else:

    bound = poly_.length
         
    for i in np.arange(0, bound + 0.1, step):
        p0 = poly_.interpolate(i, normalized=(step < 1 and step is not None))
        re.append(np.array(p0))
    
    re = np.array(re)
    if n_samples is not None:    
        return re[:n_samples]
    else:
        return re


def angle_between(v1, v2):
    """
    Return the angle between 2 vectors ( < np.pi)
    :param v1:
    :param v2:
    :return:
    """
    a = abs(atan3(v1[0], v1[1]) - atan3(v2[0], v2[1]))
    if a > np.pi:
        return 2 * np.pi - a
    else:
        return a
    # return math.acos( v1.dot(v2) / (norm(v1) * norm(v2)) )


# #TODO: might return incorrect result!
def angle_between1(vat, wat):
    """
    :param vat:
    :param wat:
    :return:
    """
    vat = np.asfarray(vat)
    wat = np.asfarray(wat)
    #vat[0] += 1e-10
    #wat[0] += 1e-10
    
    if vat[0] == 0 and wat[0] != 0:
        return np.pi/2. - abs(math.atan(wat[1] / wat[0]))
        
    if wat[0] == 0 and vat[0] != 0:
        return np.pi/2. - abs(math.atan(vat[1] / vat[0]))
    
    if wat[0] == 0 and vat[0] == 0:
        return 0
        
    #return math.atan((tan1 - tan2) / (1 + tan1 * tan2))
    return math.atan((vat[1] / vat[0] - wat[1]/ wat[0]) / (1 + (vat[1] / vat[0]) * (wat[1]/ wat[0])))


def simplify(graph, tol=5, interpolation=10):
    '''
    simplify a graph by removing close points and 3-degree nodes
    '''
    _name = lambda x, y: str(x) + '.' + str(y)
    skipped = []
    it = list(graph.degree_iter())
    for node, degree in it:
        if degree >= 3:
            nbs = list(nx.all_neighbors(graph, node))
            for d in range(degree):
                if nx.degree(graph, nbs[d]) > 1:
                    graph.add_node(_name(node, d))
                    graph.node[_name(node, d)]['pos'] = graph.node[node]['pos']
                    graph.add_edge(nbs[d], _name(node, d))
                else:
                    graph.remove_node(nbs[d])
            
            graph.remove_node(node)
            
    subgraphs = nx.connected_component_subgraphs(graph)
    newgraph = nx.Graph()
    
    idd = 0
    for sgraph in subgraphs:
        points = []
        count = 0
        #if sgraph.number_of_nodes() <= 2:
        #    continue
        terms = [None, None]
        for node, degree in sgraph.degree_iter():
            if degree == 1:
                terms[count] = node
                count += 1
                
        points.append(sgraph.node[terms[0]]['pos'])
        path = list(nx.all_simple_paths(sgraph, terms[0], terms[1]))[0]
        polyline = [sgraph.node[n]['pos'] for n in path]
        
        for px, p in enumerate(polyline):
            done = False
                
            if not done:
                if px > 0 and px < len(polyline) - 1:
                    if LineString([points[-1], polyline[px + 1]]) \
                    .distance(Point(p)) > tol:
                        points.append(p)
                        done = True
                        
                    if not done:
                        for sk in skipped:
                            if LineString([points[-1], polyline[px + 1]]) \
                            .distance(Point(sk)) > tol:
                                points.append(p)
                                done = True
                                break
                        
            if not done:
                skipped.append(p)
            else:
                skipped = []
        
        points.append(sgraph.node[terms[1]]['pos'])
        if interpolation != 0:
            points = sample_polyline(points, interpolation)
        #points = polyline
        idd += 1
        newgraph.add_nodes_from([(_name(idd, px), dict(pos=p)) for px, p in enumerate(points)])
        newgraph.add_path([_name(idd, px) for px in range(len(points))])
    
    return newgraph
            
def smoothen(polyline, caps, tol=11):
    '''
    smoothen a polyline by matching it with a contours
    '''
    #contours = asMultiLineString(outline)
    points = []
    caps_ = np.ravel(caps) #reduce(lambda x, y: x + y, caps)
    newcaps = []
    skipped = []
    for px, p in enumerate(polyline):
        done = False
        for c in caps_:
            if px == c:
                newcaps.append(len(points))
                points.append(p)
                done = True
                break
            
        if not done:
            if px > 0 and px < len(polyline) - 1:
                if LineString([points[-1], polyline[px + 1]]) \
                .distance(Point(p)) > 5:
                    points.append(p)
                    done = True
                    
                if not done:
                    for sk in skipped:
                        if LineString([points[-1], polyline[px + 1]]) \
                        .distance(Point(sk)) > 5:
                            points.append(p)
                            done = True
                            break
                    
        if not done:
            skipped.append(p)
        else:
            skipped = []
        '''            
        if not done:
            for contour in outline:
                if done: break
                for v in contour:
                    if euc(v, p) <= tol:
                        points.append(p)
                        done = True
                        break
                        
        '''
        

    newcaps = [[newcaps[0], newcaps[3]], [newcaps[1], newcaps[2]]]
    #points = [np.array(p) for p in points]
    return points, newcaps

def curvatures(rs,es):
    center = len(rs) / 2
    r = np.zeros((3,2))
    vecs, vals = [] , []
    r[1] = rs[center]
    for i in range(0,center):
        r[0] = rs[i]
        r[2] = rs[len(rs) - i - 1]
        val, vec = es.curvature(r)
        vals.append(val)
        vecs.append(vec)
    return vals, vecs

def min_curvature(rs,es):
    vals, vecs = curvatures(rs,es)
    mx = np.argmin(vals)
    return vals[mx], vecs[mx]
    
def ave_normal(rs, es):
    '''
    find averaged normal vector in a neighborhood
    '''
    if len(rs) == 1:
        return np.array([-rs[0][1], rs[0][0]])
    
    if len(rs) == 2:
        z = (rs[1] - rs[0]) / norm(rs[1] - rs[0])
        return np.array([-z[1], z[0]])

    if len(rs) % 2 == 0:
        rs = rs[:-1]
        
    center = len(rs) / 2
    r = np.zeros((3,2))
    try:
        r[1] = rs[center]
    except IndexError:
        print('IndexError')
        
    n = es.normal(np.array([rs[0],r[1],rs[-1]]))
    for i in range(1,center):
        r[0] = rs[i]
        r[2] = rs[len(rs) - i - 1]
        n = (n + es.normal(r)) / 2. #np.cross(n, es.normal(r) )
    return n / norm(n)
    
def ave_tangent(rs, es):
    '''
    find averaged normal vector in a neighborhood
    '''
    if len(rs) == 1:
        return np.array(rs[0]/norm(rs[0]))
    if len(rs) == 2:
        if norm(rs[1] - rs[0]) < 1e-5:
            return None
        z = (rs[1] - rs[0]) / norm(rs[1] - rs[0])
        return z
        
    #check if the points are too close
    close = True
    for i in rs:
        for j in rs[1:]:
            if not np.allclose(i, j, atol=1e-1):
                close = False
                break
    if close:
        return None
        
    center = len(rs) / 2
    r = np.zeros((3,2))
    r[1] = rs[center]
    t = es.tangent(np.array([rs[0],r[1],rs[-1]]))
    for i in range(1,center):
        r[0] = rs[i]
        r[2] = rs[len(rs) - i - 1]
        t = (t + es.tangent(r)) / 2. #np.cross(n, es.normal(r) )
    return t / norm(t)


class Ddiff:
    @staticmethod   
    def curvature(r):
        '''
        calculate the curavature of a polyline at point 'center'
        '''
        b = euc(r[2] , r[1])
        a = euc(r[1] , r[0])
        vec = 2 * r[0] / (a * (a + b)) - 2 * r[1] / (a * b) + 2 * r[2] / ((a + b) * b)
        val = norm(vec)
        return val, vec

    @staticmethod    
    def tangent(r):
        '''
        calculate the tangent of a polyline at point 'center'
        '''
        b = euc(r[2] , r[1])
        a = euc(r[1] , r[0])
        return (r[2] - r[1]) / b  + (r[1] - r[0]) / a  - (r[2] - r[0]) / (a + b)
    @staticmethod    
    def normal(r):
        t = Ddiff.tangent(r)
        return np.array([-t[1], t[0]])

'''
@see Asymtotic Analysis of Three Point Approximations of Vertex Normals and Curvatures, 
Elena V. Anoshkina et. al 2002
'''

class Angle:
    @staticmethod
    def curvature(r):
        a = norm(r[1] - r[0])
        b = norm(r[2] - r[1])
        va = r[1] - r[0]
        vb = r[2] - r[1]
        
        cs = dot(va, vb) / (a * b)
        cs = np.round(cs, decimals=8)
        ang = math.acos(cs)
        val = 2 * ang / (a + b)
        return val, Angle.normal_(r) * val
        
    @staticmethod    
    def normal_(r):
        a = norm(r[1] - r[0])
        b = norm(r[2] - r[1])        
        va = r[1] - r[0]
        vb = r[2] - r[1]
        at = np.array([-va[1], va[0]])
        bt = np.array([-vb[1], vb[0]])
        return 0.5 * (at/a + bt/b)
        
    @staticmethod
    def tangent(r):
        n = Angle.normal(r)
        return np.array([n[1],-n[0]])
        
    @staticmethod
    def normal(r):
        k,_ = Angle.curvature(r)
        a = max(norm(r[1] - r[0]), eps)
        b = max(norm(r[2] - r[1]), eps)
        va = r[1] - r[0]
        vb = r[2] - r[1]
        at = np.array([-va[1], va[0]])
        bt = np.array([-vb[1], vb[0]])
        return (a * b) / (a + b) * (at / a**2 + bt / b**2) * (1 - (a * b * k**2) / 8.)**-1

def load_svg_font(fontname, chars):
    import svgfig
    #from svg.path import parse_path    
    from PyQt4.QtGui import QPainterPath
    import re
    
    svg = svgfig.load(fontname)
    #select the first group as the brush
    all_chars = []
    #all_tags = []
    for subx, sub in enumerate(svg.sub[0][0].sub):
        if sub.t == 'glyph':
            if not sub['unicode'] in chars:
                continue
            
            #glyph_tags = []
            data = sub['d']
            contours = data.split('Z')
            #path = QPainterPath()
            paths = []
            for contour in contours:
                #tags = []
                path = QPainterPath()
                
                path.tags = []
                path.verts = []
                
                elems = re.split('[A-Z]+', contour)
                elems = [e for e in elems if e != '']
                if len(elems) == 0:
                    break
                
                codes = re.findall('[A-Z]+', contour)
                all_coords = [[int(num) for num in elem.split(' ')] for elem in elems]
                assert(len(codes) == len(elems))
                prev_coords = None
                current_coords = all_coords[0][0:2]
                for cx in range(len(codes)):
                    code = codes[cx]
                    elem = elems[cx]
                    coords = all_coords[cx]
                    
                    if code == 'M':
                        path.moveTo(coords[0], coords[1])
                        path.tags.append(True)
                    elif code == 'L':
                        path.lineTo(coords[0], coords[1])
                        path.tags.extend([True, True])
                    elif code == 'H':
                        path.lineTo(coords[0], current_coords[1])
                        path.tags.extend([True, True])
                    elif code == 'V':
                        path.lineTo( current_coords[0], coords[0])     
                        path.tags.extend([True, True])
                    elif code =='Q':
                        prev_coords = [coords[0], coords[1]]
                        path.quadTo(coords[0], coords[1], coords[2], coords[3])
                        path.tags.extend([True, False, True])
                    elif code =='T':
                        if not prev_coords is None:
                            x1, y1 = current_coords[0] - prev_coords[-2] + current_coords[0], \
                            current_coords[1] - prev_coords[-1] + current_coords[1]       
                            prev_coords = [x1, y1]
                            
                        path.quadTo(x1, y1, coords[0], coords[1])
                        path.tags.extend([True, False, True])
                        
                    current_coords = coords[-2:]
                paths.append(path)
            all_chars.append(paths)
    return all_chars
            
def to_paths(fontname, chars):
    from PyQt4.QtGui import QPainterPath
    allchars = []
    for ch in chars:
        paths = []
        face = Face(fontname)
        face.set_char_size( 32 * 64 )
        face.load_char(ch)

        slot = face.glyph
        outline = slot.outline
        points = np.array(outline.points, dtype=[('x',float), ('y',float)])

        start, end = 0, 0

        # Iterate over each contour
        for i in range(len(outline.contours)):
            path = QPainterPath()
            end    = outline.contours[i]
            points = outline.points[start:end+1] 
            points.append(points[0])
            tags   = outline.tags[start:end+1]
            tags.append(tags[0])
    
            segments = [ [points[0],], ]
            for j in range(1, len(points) ):
                #print "{0:b}".format(tags[j])
                segments[-1].append(points[j])
                onpath = bool(tags[j] & (1 << 0))
                order = 3 if (tags[j] & (1 << 1)) >> 1 else 2
                #print onpath, order, points[j]
                if tags[j] & (1 << 0) and j < (len(points)-1):
                    segments.append( [points[j],] )
                    
            #verts = [points[0], ]
            #codes = [Path.MOVETO,]
            path.moveTo(*points[0])
            
            for segment in segments:
                print(len(segment))
                if len(segment) == 2:
                    #verts.extend(segment[1:])
                    #codes.extend([Path.LINETO])
                    path.lineTo(*segment[1])
                elif len(segment) == 3:
                    #verts.extend(segment[1:])
                    #codes.extend([Path.CURVE3, Path.CURVE3])
                    path.quadTo(segment[1][0], segment[1][1], segment[2][0],segment[2][1])
                elif len(segment) == 4:
                    path.cubicTo(segment[1][0],segment[1][1],segment[2][0],segment[2][1],segment[3][0],segment[3][1])
                else:
                    #verts.append(segment[1])
                    #codes.append(Path.CURVE3)
                    
                    for i in range(1,len(segment)-2):
                        A,B = segment[i], segment[i+1]
                        C = ((A[0]+B[0])/2.0, (A[1]+B[1])/2.0)
                        #verts.extend([ C, B ])
                        #codes.extend([ Path.CURVE3, Path.CURVE3])
                        path.quadTo(C[0], C[1], B[0], B[1])                        

            start = end+1
            paths.append(path)

        allchars.append(paths)

    return allchars        

def to_polygons(fontname,chars):
    allchars = []
    for ch in chars:
        face = Face(fontname)
        face.set_char_size( 32 * 64 )
        face.load_char(ch)
        '''
        bitmap = face.glyph.bitmap
        width  = face.glyph.bitmap.width
        rows   = face.glyph.bitmap.rows
        pitch  = face.glyph.bitmap.pitch
        '''
        slot = face.glyph
    
        outline = slot.outline
        points = np.array(outline.points, dtype=[('x',float), ('y',float)])
        #x, y = points['x'], points['y']
        start, end = 0, 0
        VERTS, CODES = [], []
        # Iterate over each contour
        for i in range(len(outline.contours)):
            end    = outline.contours[i]
            points = outline.points[start:end+1] 
            points.append(points[0])
            tags   = outline.tags[start:end+1]
            tags.append(tags[0])
    
            segments = [ [points[0],], ]
            for j in range(1, len(points) ):
                segments[-1].append(points[j])
                if tags[j] & (1 << 0) and j < (len(points)-1):
                    segments.append( [points[j],] )
            verts = [points[0], ]
            codes = [Path.MOVETO,]
            for segment in segments:
                if len(segment) == 2:
                    verts.extend(segment[1:])
                    codes.extend([Path.LINETO])
                elif len(segment) == 3:
                    verts.extend(segment[1:])
                    codes.extend([Path.CURVE3, Path.CURVE3])
                else:
                    verts.append(segment[1])
                    codes.append(Path.CURVE3)
                    for i in range(1,len(segment)-2):
                        A,B = segment[i], segment[i+1]
                        C = ((A[0]+B[0])/2.0, (A[1]+B[1])/2.0)
                        verts.extend([ C, B ])
                        codes.extend([ Path.CURVE3, Path.CURVE3])
                    verts.append(segment[-1])
                    codes.append(Path.CURVE3)
            VERTS.extend(verts)
            CODES.extend(codes)
            start = end+1
    
        # Draw glyph
        path = Path(VERTS, CODES)
        poly = path.to_polygons()
        #print zzz.to_polygons()
        '''
        newpoly = [] #poly
        for p in poly:
            newp = Path(p,[Path.MOVETO] + ([Path.LINETO] * (len(p)-1))).interpolated(steps=1).to_polygons()[0]
            newpoly.append(newp)
        '''
        #plt.scatter(poly[0][:,0],poly[0][:,1])
        #plt.show()
        
        allchars.append(poly)
    #if len(allchars) == 1:
    #    return allchars[0]
    #else:
    return allchars

def smooth_worker(i, files):
    for f in files:
        print(f)
        data = np.load(f)
        anns = data['annotations']
        segs = data['segments']
        caps = data['caps']
        outlines = data['outlines']                
        for c in range(26, 52):
            nsegs = []
            ncaps = []
            for p in range(len(anns[c])):
                nseg, ncap = smoothen(outlines[c], segs[c][p], caps[c][p], 5)
                nsegs.append(nseg)
                ncaps.append(ncap)
            segs[c] = nsegs
            caps[c] = ncaps
            
        modify_npz(f, [('segments', segs), ('caps', caps)])
    
if __name__ == '__main__':
    from Common import modify_npz
    import os
    from os.path import join
    import multiprocessing
    #import geometry as geo
    #print geo.ave_normal(np.ascontiguousarray(np.array([[1,2],[10,20],[8,7]]), dtype='f'))
    path = '/media/phan/BIGDATA/SMARTFONTS/data'
    all_chars = list("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890")
    files = sorted([f for f in os.listdir(path) \
            if os.path.isfile(join(path, f)) and f.endswith('.npz')])
    files_ = []
    specs = np.load(join(path, 'word_specs.npy'))
    for f in files:
        data = np.load(join(path, f))
        if 'segments' in data and 'annotations' in data:
            segs = data['segments']
            specs_ = np.array([len(a) for a in segs])
            if np.all(specs_ >= specs):
                #print f
                files_.append(join(path, f))
                
    step = len(files_)/4
    print('processing', len(files_), 'files')
    for i in range(4):
        if i == 3:
            last = len(files_)
        else:
            last = step
            
        p = multiprocessing.Process(target=smooth_worker, args=(10, files_[i * step : i * step + last]))
        p.start()
        
    