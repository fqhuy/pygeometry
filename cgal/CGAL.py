import cgal_straight_skeleton as sk
import numpy as np
from shapely.geometry import *
import networkx as nx
from scipy.spatial.distance import euclidean as euc

eps = 1e-10
BISECTOR = 1
CONTOUR = 2
INNER_BISECTOR = 0

SKELETON = 1
NON_SKELETON = 0


def intersection(p1, p2, p3, p4):
    '''
    between 2 line segments
    '''
    p1, p2, p3, p4 = (np.asfarray(p1),np.asfarray(p2),np.asfarray(p3),np.asfarray(p4))
    denom = ((p4[0] - p3[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (p4[1] - p3[1]))
    if denom != 0:
        ta = ((p3[1] - p4[1]) * (p1[0] - p3[0]) + (p4[0] - p3[0]) * (p1[1] - p3[1])) \
        / denom
        tb = ((p1[1] - p2[1]) * (p1[0] - p3[0]) + (p2[0] - p1[0]) * (p1[1] - p3[1])) \
        / denom
        
        if 0 <= ta <= 1 and 0 <= tb <= 1:
            return p1 + ta * (p2 - p1)
        else:
            return None
    else:
        return None


def extract_skeleton(polylines, step=35):
    """
    extract skeleton, auto detect holes
    :param polylines: a collection of polyline
    :param step
    """
    _name = lambda x, y: str(x) + '.' + str(y)
    holes = []
    polylines = []
    
    for i in range(len(polylines_)):
        try:
            thisone = asPolygon(polylines_[i])
            valid = thisone.is_valid
            polylines.append(polylines_[i])
        except ValueError as e:
            pass

    # contours = asMultiLineString(polylines)
    for i in range(len(polylines)):
        thisone = asPolygon(polylines[i])
        if not thisone.is_valid:
            print '#', 
            thisone = thisone.buffer(0)
            newply = []
            if not isinstance(thisone, Polygon):
                return None

            if thisone.exterior is None:
                #print '*'
                return None
                    
            for crd in thisone.exterior.coords:
                newply.append(np.array(crd))
            polylines[i] = np.asarray(newply, dtype='float64')
        
        is_hole = False
        outer = None
        for j in range(len(polylines)):
            if i != j:
                another = asPolygon(polylines[j])
                #try:
                if not another.is_valid:
                    another = another.buffer(0)
                
                if another.contains(thisone):
                    is_hole = True
                    outer = j
                    break
                
        if is_hole:
            holes.append((i, outer))
            
    holes = np.array(holes)
    g = nx.Graph()
    vnode_count = 0
    for i in range(len(polylines)):
        holes_ = [polylines[h[0]][::-1] for h in holes if h[1] == i]
        if len(holes) != 0:
            if i in holes[:,0]:
                continue
            
        edges, emodes, vertices, vmodes = sk.straight_skeleton(polylines[i][::-1], holes_)
        edges = [[str(e[0]), str(e[1])] for ex, e in enumerate(edges) if ex % 2 == 0]
        emodes = [e for ex, e in enumerate(emodes) if ex % 2 == 0]
        vmodes = list(vmodes)
        vertices_d = dict(zip([str(int(v)) for v in vertices[:, 2]], vertices[:, 0:2]))
        vertices = [[ v[0], v[1], str(int(v[2]))] for v in vertices]
        inner_edges = [[edges[e][0], edges[e][1]] for e in range(len(edges)) if emodes[e] == INNER_BISECTOR]
        contour_edges = [[vertices_d[edges[e][0]], vertices_d[edges[e][1]]] \
                   for e in range(len(edges)) if emodes[e] == BISECTOR]
        
        todelete = []
        for p in range(len(edges)):
            d = euc(vertices_d[edges[p][1]], vertices_d[edges[p][0]])
            if emodes[p] != CONTOUR:
                continue
            
            if d / step > 2:
                move = (vertices_d[edges[p][1]] - vertices_d[edges[p][0]]) / d
                normal = np.array([-move[1], move[0]])
                move *= step
                curr = vertices_d[edges[p][0]].copy()
                todelete.append(edges[p])
                prev_node = edges[p][0]
                saved_vnode_count = vnode_count
                while euc(curr, vertices_d[edges[p][1]]) > step:
                    curr += move
                    inters = []
                    for ex, e in enumerate(inner_edges):
                        if e in todelete:
                            continue
   
                        inter = intersection(vertices_d[e[0]], vertices_d[e[1]], curr - normal * 250, curr) #, curr + normal * 250
                        if not inter is None:
                            inters.append((inter, e))
                            
                    if len(inters) == 0:
                        continue
                    else:
                        inter, e = sorted([(j, e) for j, e in inters], key=lambda x: euc(x[0], curr))[0]
                        
                    valid = True
                    for ce in contour_edges:
                        inter_ = intersection(ce[0], ce[1], curr, inter)
                        if not inter_  is None:
                            if np.any(inter_ != ce[0]) and np.any(inter_ != ce[1]):
                                valid = False
                                break
                    if valid:
                        vnode_count += 1
                        ''' add CONTOUR nodes '''
                        newnode = 'v.' + str(vnode_count)
                        vertices.append([curr[0], curr[1], newnode])
                        vmodes.append(NON_SKELETON)
                        vertices_d[newnode] = curr.copy()
                        edges.append([newnode, prev_node])
                        emodes.append(CONTOUR)
                        prev_node = newnode
                        ''' add BISECTORs '''
                        #for _i in [0, 1]:
                        if euc(inter, vertices_d[e[0]]) < 1:
                            #print 'merging ', e[0], prev_node
                            edges.append([e[0], prev_node])
                            emodes.append(BISECTOR)
                            continue
                        
                        if euc(inter, vertices_d[e[1]]) < 1:
                            #print 'merging ', e[0], prev_node
                            edges.append([e[1], prev_node])
                            emodes.append(BISECTOR)
                            continue
                        
                        todelete.append(e)
                        newnode = 'm.' + str(vnode_count)
                        vertices.append([inter[0], inter[1], newnode])
                        vmodes.append(SKELETON)
                        edges.append([e[0], newnode])
                        emodes.append(INNER_BISECTOR)
                        edges.append([newnode, e[1]])
                        emodes.append(INNER_BISECTOR)
                        edges.append([newnode, prev_node])
                        emodes.append(BISECTOR)
                    
                        inner_edges.append([e[0], newnode])
                        inner_edges.append([newnode, e[1]])
                        vertices_d[newnode] = inter
                        #break
                if saved_vnode_count == vnode_count:
                    todelete.pop()
                else:            
                    edges.append([prev_node, edges[p][1]])
                    emodes.append(CONTOUR)
        
        newedges = []
        newemodes = []        
        for e in range(len(edges)):
            if not edges[e] in todelete:
                newedges.append(edges[e])
                newemodes.append(emodes[e])
        
        edges = newedges
        emodes = newemodes
        
        for p in range(len(edges)):
            d = euc(vertices_d[edges[p][1]], vertices_d[edges[p][0]]) 
            g.add_edge(_name(i, edges[p][0]), _name(i, edges[p][1]), d=d, mode=emodes[p])
            g.node[ _name(i, edges[p][1]) ]['pos'] = vertices_d[str(edges[p][1])]
            g.node[_name(i, edges[p][0]) ]['pos'] = vertices_d[str(edges[p][0])]
            
        for v in range(len(vertices)):
            if g.has_node(_name(i, vertices[v][2])):
                g.node[ _name(i, vertices[v][2]) ]['mode'] = vmodes[v]
 
    return g

if __name__ == '__main__':
    import argparse
    from os.path import join
    import os

    parser = argparse.ArgumentParser(description='A commandline to extract skeletons from polygons')
    parser.add_argument('files', type=str, nargs='*', help='data files, must contain the polygons')
    args = parser.parse_args()

    for fl in args.files:
        data = np.load(fl)
        graph = extract_skeleton(data)
        path, fname = os.path.split(fl)
        np.save(join(path, 'graph_' + fname), graph)


