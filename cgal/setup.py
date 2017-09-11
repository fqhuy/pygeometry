from distutils.core import setup, Extension
import numpy
import sys
# define the extension module

if sys.platform == "darwin":
    straight_skeleton = Extension('cgal_straight_skeleton', sources=['straight_skeleton.cpp'],
                              include_dirs=[numpy.get_include(), '/usr/local/include'],
    #                         include_dirs = ['/usr/local/include'],
                              libraries = ['CGAL_Core','mpfr','boost_thread-mt'],
                              library_dirs = ['/usr/local/lib','/usr/local/Cellar/boost/1.58.0/lib'],
                              )
elif sys.platform == "linux":
    straight_skeleton = Extension('cgal_straight_skeleton', sources=['straight_skeleton.cpp'],
                              include_dirs=[numpy.get_include(), '/usr/local/include'],
    #                         include_dirs = ['/usr/local/include'],
                              libraries = ['CGAL_Core','mpfr','boost_thread'],
                              library_dirs = ['/usr/local/lib'],
                              )

# run the setup
setup(ext_modules=[straight_skeleton])
