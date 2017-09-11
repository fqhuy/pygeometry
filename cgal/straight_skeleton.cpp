#include<boost/shared_ptr.hpp>

#include<CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include<CGAL/Polygon_2.h>
#include<CGAL/create_straight_skeleton_2.h>
#include<CGAL/Polygon_with_holes_2.h>
#include<CGAL/create_straight_skeleton_from_polygon_with_holes_2.h>
//#include "print.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <cstdio>
#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K ;
typedef K::Point_2                   Point ;
typedef CGAL::Polygon_2<K>           Polygon_2 ;
typedef CGAL::Polygon_with_holes_2<K> Polygon_with_holes ;
typedef CGAL::Straight_skeleton_2<K> Ss ;
typedef boost::shared_ptr<Ss> SsPtr ;

/* mode = 0: inner bisector
 * mode = 2: contour 
 * mode = 3: bisector
 */ 
template<class K>
void extract_vertices( CGAL::Straight_skeleton_2<K> const& ss , PyObject** edges, PyObject** e_modes, PyObject** vertices, PyObject** v_modes)
{
    typedef CGAL::Straight_skeleton_2<K> Ss ;

    typedef typename Ss::Vertex_const_handle     Vertex_const_handle ;
    typedef typename Ss::Vertex_const_iterator   Vertex_const_iterator ;
    typedef typename Ss::Halfedge_const_handle   Halfedge_const_handle ;
    typedef typename Ss::Halfedge_const_iterator Halfedge_const_iterator ;

    Halfedge_const_handle null_halfedge ;
    Vertex_const_handle   null_vertex ;

    int n_edges = ss.size_of_halfedges();
    long int dims[] = {n_edges, 2};
    long int dims_1[] = {n_edges};
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_INT);
    (*edges) = PyArray_Zeros(2, dims, descr, 0);
    (*e_modes) = PyArray_Zeros(1, dims_1, descr, 0);
    
    if ((*edges) == NULL || (*e_modes) == NULL)
        return;
        
    //printf("accessing memory");
    Py_INCREF(descr);
    int **c_array;
    PyArray_AsCArray(edges, (void *) &c_array, dims, 2, descr);
    int *mc_array;
    PyArray_AsCArray(e_modes, (void *) &mc_array, dims_1, 1, descr);
    //printf("done preparing memory");

    int counter = 0;
    for (Halfedge_const_iterator i = ss.halfedges_begin(); i != ss.halfedges_end(); ++i)
    {
        c_array[counter][0] = i->vertex()->id();
        c_array[counter][1] = i->opposite()->vertex()->id();
        
        if (i->is_inner_bisector()){
            mc_array[counter] = 0;
        } else if (i->is_bisector()) {
            mc_array[counter] = 1;
        } else {
            mc_array[counter] = 2;
        }
        
        ++counter;
    }
    
    int n_vertices = ss.size_of_vertices();
    long int dims_2[] = {n_vertices, 3};
    PyArray_Descr *descr_2 = PyArray_DescrFromType(NPY_DOUBLE);
    (*vertices) = PyArray_ZEROS(2, dims_2, NPY_DOUBLE, 0);
    double **v_array;
    PyArray_AsCArray(vertices, (void *) &v_array, dims_2, 2, descr_2);
    counter = 0;
    
    PyObject* v_modes_l =  PyList_New(n_vertices);
    
    for( Vertex_const_iterator i = ss.vertices_begin(); i != ss.vertices_end(); ++i){
        v_array[counter][0] = i->point().x();
        v_array[counter][1] = i->point().y();
        v_array[counter][2] = i->id();
        /*
        PyList *p = PyList_New(2);
        PyList_SetItem(p, 0, i->point().x();
        PyList_SetItem(p, 1, i->point().y();
        PyList_SetItem(v_modes_l, counter, p);
        */
        if(i->is_skeleton() ){
            PyList_SetItem(v_modes_l, counter, PyInt_FromLong(1));
        } else {
            PyList_SetItem(v_modes_l, counter, PyInt_FromLong(0));
        }
        
        ++counter;
    }
    
    (*v_modes) = PyArray_FromObject(v_modes_l, NPY_INT, 1, 2);
}

int get_c_array(PyArrayObject** obj, double ***c_array)
{
    int n_vertices = PyArray_DIM((PyObject*)(*obj), 0);
    long int dims[] = {n_vertices, 2};
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);    
    PyArray_AsCArray((PyObject**)obj, (void*)c_array, dims, 2, descr);
    return n_vertices;
}

static PyObject* straight_skeleton(PyObject* self, PyObject* args)
{
    PyArrayObject *in_array;
    //PyArrayObject *holes;
    PyListObject *holes;
    PyObject *edges;
    PyObject *vertices;
    PyObject *e_modes;
    PyObject *v_modes;
   
    //if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &in_array,  &PyArray_Type, &holes))
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &in_array,  &PyList_Type, &holes))
        return NULL;

    int n_holes = PyList_Size((PyObject*)holes);
    bool with_hole = n_holes > 0;
        
    double **c_array;
    int n_vertices = get_c_array(&in_array, &c_array);
    
    double x, y;
    Polygon_2 outer ;
    for(int i = 0; i < n_vertices; ++i){
        x = c_array[i][0];
        y = c_array[i][1];
        outer.push_back(Point(x, y));
    }

    Polygon_with_holes poly(outer) ; 
 
    if(with_hole == true){
        for(int h = 0; h < n_holes; ++h){
            Polygon_2 poly_hole;
            PyArrayObject *tmp = (PyArrayObject*) PyList_GetItem((PyObject*)holes, h);
            n_vertices = get_c_array(&tmp, &c_array);
            for(int i = 0; i < n_vertices; ++i){
                x = c_array[i][0];
                y = c_array[i][1];
                poly_hole.push_back(Point(x, y));
            }
            poly.add_hole( poly_hole ) ;
        }
    }
    //printf("begin skeletonize..");
    SsPtr iss = CGAL::create_interior_straight_skeleton_2(poly);
    //printf("extracting vertices");
    extract_vertices(*iss, &edges, &e_modes, &vertices, &v_modes);

    PyObject *t;

    t = PyTuple_New(4);
    PyTuple_SetItem(t, 0, edges);
    PyTuple_SetItem(t, 1, e_modes);
    PyTuple_SetItem(t, 2, vertices);
    PyTuple_SetItem(t, 3, v_modes);

    Py_INCREF(t);

    return t;
}

/*  define functions in module */
static PyMethodDef Methods[] =
{
     {"straight_skeleton", straight_skeleton, METH_VARARGS,
         "compute the straight skeleton of a shape"},
     {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC
initcgal_straight_skeleton(void)
{
     (void) Py_InitModule("cgal_straight_skeleton", Methods);
     /* IMPORTANT: this must be called */
     import_array();
}
