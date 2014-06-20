Intrinsic Images in the Wild
============================

This repository contains the decomposition algorithm presented in our
[paper](http://intrinsic.cs.cornell.edu):

	Sean Bell, Kavita Bala, Noah Snavely
	"Intrinsic Images in the Wild"
	ACM Transactions on Graphics (SIGGRAPH 2014)

	@article{bell14intrinsic,
		author = "Sean Bell and Kavita Bala and Noah Snavely",
		title = "Intrinsic Images in the Wild",
		journal = "ACM Trans. on Graphics (SIGGRAPH)",
		volume = "33",
		number = "4",
		year = "2014",
	}

as well as a simple Python wrapper to the C++ dense CRF inference code from
[Krahenbuhl et al 2013](http://graphics.stanford.edu/projects/drf/):

	Philipp Krähenbühl and Vladlen Koltun
    "Parameter Learning and Convergent Inference for Dense Random Fields"
    International Conference on Machine Learning (ICML) 2013

It's nice to see how many people are using our code, so please "star" our
repository on Github.

The dataset is hosted at http://intrinsic.cs.cornell.edu/.


Dependencies
------------

The following libraries are needed:

- Eigen (http://eigen.tuxfamily.org/)
  On Ubuntu, you can install with: `sudo apt-get install libeigen3-dev`

- Python 2.7

- Python packages (newer packages will likely work, though these are the exact
  versions that I used):

      PIL==1.1.7
      cython==0.19.2
      numpy==1.8.0
      scipy==0.13.2
      scikit-image==0.9.3
      scikit-learn==0.14.1


Compile
-------

If on Ubuntu and you have installed Eigen3 to its default directory (/usr/include/eigen3),
then you can build the C++ extension with:

    cd krahenbuhl2013/
    make

If you are on another operating system or `eigen3` is in another directory,
edit `krahenbuhl2013/setup.py` to change the directory.


Running
-------

Basic usage:

    bell2014/decompose.py image.png

All arguments:

    usage: decompose.py [-h] [-r <file>] [-s <file>] [-m <file>] [-j <file>]
                        [-p <file>] [-l] [-q] [--show-labels]
                        <file>

    Decompose an image using the algorithm presented in: Sean Bell, Kavita Bala,
    Noah Snavely. "Intrinsic Images in the Wild". ACM Transactions on Graphics
    (SIGGRAPH 2014). http://intrinsic.cs.cornell.edu. The output is rescaled for
    viewing and encoded as sRGB PNG images(unless --linear is specified).

    positional arguments:
      <file>                Input image

    optional arguments:
      -h, --help            show this help message and exit
      -r <file>, --reflectance <file>
                            Reflectance layer output name (saved as sRGB image)
      -s <file>, --shading <file>
                            Shading layer output name (saved as sRGB image)
      -m <file>, --mask <file>
                            Mask filename
      -j <file>, --judgements <file>
                            Judgements file from the Intrinsic Images in the Wild
                            dataset
      -p <file>, --parameters <file>
                            Parameters file (JSON format). See bell2014/params.py
                            for a list of parameters.
      -l, --linear          if specified, assume input is linear and generate
                            linear output, otherwise assume input is sRGB and
                            generate sRGB output
      -q, --quiet           if specified, don't print logging info
      --show-labels         if specified, also output labels


Tone-mapping
------------

All input images are assumed to be sRGB, and the output reflectance and shading
layers are tone-mapped to sRGB.  If using linear images (e.g., the MIT
Intrinsic Images Dataset, http://www.cs.toronto.edu/~rgrosse/intrinsic/),
specify `--linear` and both input/output will remain linear.


Embedding in other projects
---------------------------

Our code was written to be modular and can be embedded in larger projects.
The basic code for constructing the input and parameters can be found in
`bell2014/decompose.py` or is listed below:

```python
from bell2014.solver import IntrinsicSolver
from bell2014.input import IntrinsicInput
from bell2014.params import IntrinsicParameters
from bell2014 import image_util

input = IntrinsicInput.from_file(
	image_filename,
	image_is_srgb=sRGB,
	mask_filename=mask_filename,
	judgements_filename=judgements_filename,
)

params = IntrinsicParameters.from_dict({
	'param1': ...,
	'param2': ...,
})

solver = IntrinsicSolver(input,params)
r, s, decomposition = solver.solve()

image_util.save(r_filename, r, mask_nz=input.mask_nz, rescale=True)
image_util.save(s_filename, s, mask_nz=input.mask_nz, rescale=True)
```
