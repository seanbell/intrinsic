import os
import math
import numpy as np
from skimage.color import rgb2lab
from skimage.transform import resize

from . import image_util
from judgements import HumanReflectanceJudgements


class IntrinsicInput(object):
    """ Input to a decomposition.  All properties are read-only. """

    def __init__(self, image_rgb, mask=None, r_gt=None, s_gt=None,
                 judgements=None, dataset=None, id=None):

        self._dataset = dataset
        self._id = id

        # convert to color
        if image_rgb.ndim == 2:
            self._image_rgb = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3))
            self._image_rgb[:, :, :] = image_rgb[:, :, np.newaxis]
        elif image_rgb.ndim == 3:
            self._image_rgb = image_rgb.copy()
        else:
            raise ValueError("Invalid image")

        # drop alpha channel
        if self._image_rgb.shape[2] == 4:
            self._image_rgb = self._image_rgb[:, :, 0:3]
        elif self._image_rgb.shape[2] != 3:
            raise ValueError("Invalid image")

        # Clip to 1e-4 since a sRGB value of 1/255 is (1/255)/12.92 ~= 3e-4
        # in linear RGB space.  This avoids a huge log-space jump between
        # intensity 0/255 and intensity 1/255.
        self._image_rgb[self._image_rgb < 1e-4] = 1e-4
        self._image_rgb.setflags(write=False)

        # load binary image mask
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("Invalid mask")
            self._mask = mask.astype(np.bool, copy=True)
        else:
            self._mask = np.ones((self.rows, self.cols), dtype=bool)
        self._mask.setflags(write=False)

        assert self.mask.shape == self.image_rgb.shape[0:2]

        # load ground truth
        if r_gt is not None:
            self._r_gt = r_gt.copy()
            self._r_gt.setflags(write=False)
        else:
            self._r_gt = None

        if s_gt is not None:
            self._s_gt = s_gt.copy()
            self._s_gt.setflags(write=False)
        else:
            self._s_gt = None

        # human judgements
        self._judgements = judgements

    @staticmethod
    def from_dataset(dataset, in_dir, id):
        """ Load input from a dataset """

        if dataset == "mit":
            return IntrinsicInput.from_file(
                image_filename=os.path.join(in_dir, id, 'diffuse.png'),
                image_is_srgb=False,
                mask_filename=os.path.join(in_dir, id, 'mask.png'),
                r_gt_filename=os.path.join(in_dir, id, 'reflectance.png'),
                s_gt_filename=os.path.join(in_dir, id, 'shading.png'),
                gt_is_srgb=False,
                dataset=dataset,
                id=id,
            )
        elif dataset == "iiw":
            return IntrinsicInput.from_file(
                image_filename=os.path.join(in_dir, '%s.png' % id),
                image_is_srgb=True,
                judgements_filename=os.path.join(in_dir, '%s.json' % id),
                dataset=dataset,
                id=id,
            )
        else:
            raise ValueError("Unknown dataset")

    @staticmethod
    def from_file(image_filename, image_is_srgb=True, mask_filename=None,
                  r_gt_filename=None, s_gt_filename=None, gt_is_srgb=False,
                  judgements_filename=None, dataset=None, id=None):
        """ Load input from files """

        image_rgb = image_util.load(image_filename, is_srgb=image_is_srgb)

        if mask_filename:
            mask = image_util.load_mask(mask_filename)
        else:
            mask = None

        if r_gt_filename:
            r_gt = image_util.load(r_gt_filename, is_srgb=gt_is_srgb)
        else:
            r_gt = None

        if s_gt_filename:
            s_gt = image_util.load(s_gt_filename, is_srgb=gt_is_srgb)
        else:
            s_gt = None

        if judgements_filename:
            judgements = HumanReflectanceJudgements.from_file(judgements_filename)
        else:
            judgements = None

        return IntrinsicInput(
            image_rgb=image_rgb, mask=mask,
            r_gt=r_gt, s_gt=s_gt,
            judgements=judgements,
            dataset=dataset, id=id,
        )

    def downsample(self, factor=2.0):
        """ Return a new input at a lower resolution (while keeping ground
        truth (gt) resolution the same). """

        if factor <= 1.0:
            return self

        rows = int(self.rows / factor)
        cols = int(self.cols / factor)
        image_rgb = resize(self.image_rgb, (rows, cols, 3), mode='reflect')

        if self.mask_nnz == self.rows * self.cols:
            mask = None
        else:
            mask = resize(self.mask.astype(float), (rows, cols), mode='reflect')

        return IntrinsicInput(image_rgb, mask, self.r_gt, self.s_gt)

    ## IMAGE TAG ##

    @property
    def id(self):
        return self._id

    @property
    def dataset(self):
        return self._dataset

    ## IMAGE MASK ##

    @property
    def mask(self):
        return self._mask

    @property
    def mask_nz(self):
        if not hasattr(self, '_mask_nz'):
            self._mask_nz = np.nonzero(self.mask)
        return self._mask_nz

    @property
    def mask_nnz(self):
        """ Shorthand for the number of nonzero entries """
        return self.mask_nz[0].size

    ## IMAGE SIZE ##

    @property
    def shape(self):
        return self._image_rgb.shape

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    @property
    def diag(self):
        """ diagonal of the inner bounding box of nonzero mask pixels """
        if not hasattr(self, '_diag'):
            if self.mask_nz:
                self._diag = math.sqrt(np.sum([
                    (np.max(nz) - np.min(nz)) ** 2
                    for nz in self.mask_nz
                ]))
            else:
                self._diag = math.sqrt(self.rows ** 2 + self.cols ** 2)
        return self._diag

    ## IMAGE COLORSPACES ##

    @property
    def image_rgb(self):
        """ Image in linear RGB space """
        return self._image_rgb

    @property
    def image_rgb_nz(self):
        """ Image linear RGB space with only unmasked (_nz = "nonzero mask")
        entries """
        return self._image_rgb[self.mask_nz]

    @property
    def image_gray(self):
        """ Image in grayscale space with only unmasked (_nz = "nonzero
        mask") entries """
        if not hasattr(self, '_image_gray'):
            self._image_gray = np.mean(self._image_rgb, axis=2)
            self._image_gray.setflags(write=False)
        return self._image_gray

    @property
    def log_image_gray(self):
        """ Image in log-grayscale space """
        if not hasattr(self, '_log_image_gray'):
            # no clip necessary since we clip at load time
            self._log_image_gray = np.log(self.image_gray)
            self._log_image_gray.setflags(write=False)
        return self._log_image_gray

    @property
    def log_image_rgb(self):
        """ Image in log(linear RGB) space """
        if not hasattr(self, '_log_image_rgb'):
            # no clip necessary since we clip at load time
            self._log_image_rgb = np.log(self.image_rgb)
            self._log_image_rgb.setflags(write=False)
        return self._log_image_rgb

    @property
    def image_gray_nz(self):
        """ Image in grayscale space with only unmasked (_nz = "nonzero
        mask") entries """
        return self.image_gray[self.mask_nz]

    @property
    def image_irg(self):
        """ Image in 'irg' space (intensity, red chromaticity, green
        chromaticity) """

        if not hasattr(self, '_image_irg'):
            self._image_irg = image_util.rgb_to_irg(self._image_rgb)
            self._image_irg.setflags(write=False)
        return self._image_irg

    @property
    def image_irg_nz(self):
        """ Image in 'irg' space with only unmasked (_nz = "nonzero
        mask") entries """

        return self.image_irg[self.mask_nz]

    @property
    def image_lab(self):
        """ Image in L*a*b* space """
        if not hasattr(self, '_image_lab'):
            self._image_lab = rgb2lab(self._image_rgb)
            self._image_lab.setflags(write=False)
        return self._image_lab

    def image(self, colorspace='rgb'):
        return getattr(self, 'image_%s' % colorspace)

    ## GROUND TRUTH ##

    @property
    def judgements(self):
        return self._judgements

    @property
    def r_gt(self):
        return self._r_gt

    @property
    def s_gt(self):
        return self._s_gt

    def compute_lmse(self, r, s, window_size=20):
        """ Compute LMSE error, as per [Grosse et al 2009] MIT Intrinsic Images
        dataset """
        r = np.mean(r, axis=-1)
        r_gt = self.r_gt
        if r_gt.ndim == 3:
            r_gt = np.mean(r_gt, axis=-1)
        from .lmse import score_image
        return score_image(
            self.s_gt, r_gt,
            s, r, self.mask, window_size)

    def compute_whdr(self, r, delta=0.10):
        return self.judgements.compute_whdr(r, delta)

    def compute_error(self, r, s):
        """ Compute error for the decomposition using the error metric
        associated with the dataset """

        if self.dataset == "mit":
            return self.compute_lmse(r, s)
        elif self.dataset == "iiw":
            return self.compute_whdr(r)
        else:
            raise ValueError("Unknown dataset: %s" % self.dataset)
