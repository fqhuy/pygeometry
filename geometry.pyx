# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 11:30:14 2014

@author: phan
"""

#cimport rfs
cimport numpy as np
import numpy as np
import math
from libc.math cimport acos
from cython cimport view

cdef double eps = 10e-10

cpdef ave_normal(np.ndarray[double, ndim=2, mode="c"] rs):
    '''
    find averaged normal vector in a neighborhood
    '''
    if len(rs) == 1:
        return np.array([-rs[0][1], rs[0][0]])
    
    cdef np.ndarray[double, ndim=1, mode="c"] z = np.zeros(2, dtype='d', order='C')
    if len(rs) == 2:
        z = (rs[1] - rs[0]) / np.linalg.norm(rs[1] - rs[0])
        return np.array([-z[1], z[0]])

    if len(rs) % 2 == 0:
        rs = rs[:-1]
        
    cdef int center = len(rs) / 2
    cdef np.ndarray[double, ndim=2, mode="c"] r = np.zeros((3,2), dtype='d', order='C')
    #try:
    r[1] = rs[center]
    #except IndexError:
    #    print 'IndexError'
        
    cdef np.ndarray[double, ndim=1, mode="c"] n = normal(np.array([rs[0],r[1],rs[-1]], dtype='d'))
    for i in range(1, center):
        r[0] = rs[i]
        r[2] = rs[len(rs) - i - 1]
        n = (n + normal(r)) / 2.
        
    return n / np.linalg.norm(n)
    
cpdef curvature(np.ndarray[double, ndim=2, mode="c"] r):
    cdef double a = np.linalg.norm(r[1] - r[0])
    cdef double b = np.linalg.norm(r[2] - r[1])
    cdef np.ndarray[double,ndim=1,mode="c"] va = r[1] - r[0]
    cdef np.ndarray[double,ndim=1,mode="c"] vb = r[2] - r[1]
    
    cdef double cs = np.dot(va, vb) / (a * b)
    cs = np.round(cs, decimals=8)
    cdef double ang = acos(cs)
    cdef double val = 2 * ang / (a + b)
    return val#, normal_(r) * val
  
cpdef normal_(np.ndarray[double, ndim=2, mode="c"] r):
    cdef double a = np.linalg.norm(r[1] - r[0])
    cdef double b = np.linalg.norm(r[2] - r[1])        
    cdef np.ndarray[double,ndim=1,mode="c"] va = r[1] - r[0]
    cdef np.ndarray[double,ndim=1,mode="c"] vb = r[2] - r[1]
    cdef np.ndarray[double,ndim=1,mode="c"] at = np.array([-va[1], va[0]], dtype='d')
    cdef np.ndarray[double,ndim=1,mode="c"] bt = np.array([-vb[1], vb[0]], dtype='d')
    return 0.5 * (at / a + bt / b)

cpdef tangent(np.ndarray[double, ndim=2, mode="c"] r):
    cdef np.ndarray[double,ndim=1,mode="c"] n = normal(r)
    return np.array([n[1],-n[0]])

cpdef normal(np.ndarray[double, ndim=2, mode="c"] r):
    cdef double k = curvature(r)
    cdef double a = max(np.linalg.norm(r[1] - r[0]), eps)
    cdef double b = max(np.linalg.norm(r[2] - r[1]), eps)
    cdef np.ndarray[double,ndim=1,mode="c"] va = r[1] - r[0]
    cdef np.ndarray[double,ndim=1,mode="c"] vb = r[2] - r[1]
    cdef np.ndarray[double,ndim=1,mode="c"] at = np.array([-va[1], va[0]], dtype='d')
    cdef np.ndarray[double,ndim=1,mode="c"] bt = np.array([-vb[1], vb[0]], dtype='d')
    cdef np.ndarray[double,ndim=1,mode="c"] re = \
    (a * b) / (a + b) * (at / a**2 + bt / b**2) * (1 - (a * b * k**2) / 8.)**-1
    re = re / np.linalg.norm(re)
    return re