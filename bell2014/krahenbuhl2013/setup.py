from distutils.core import setup
from distutils.extension import Extension
try:
    from Cython.Build import cythonize
    import numpy
except ImportError:
    print "You must have Cython >=0.17 and NumPy to build!"
    import sys
    sys.exit(1)

setup(
    ext_modules=cythonize(Extension(
        'krahenbuhl2013',
        sources=[
            'krahenbuhl2013.pyx',
            "src/densecrf.cpp",
            "src/labelcompatibility.cpp",
            "src/pairwise.cpp",
            "src/permutohedral.cpp",
            "src/unary.cpp",
            "src/util.cpp",
            "src/densecrf_wrapper.cpp",
        ],
        include_dirs=[
            numpy.get_include(),
            "include",
            "/usr/include/eigen3",
        ],
        language="c++",
    )),
)
