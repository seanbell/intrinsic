import os
import numpy as np
from . import image_util


class IntrinsicDecomposition(object):
    """ Current state of a reconstruction.  All entries (except ``input``) are
    mutable. """

    def __init__(self, params, input):
        self._input = input
        self.params = params

        # iteration number
        self.iter_num = None

        # stage 1 or 2 (each iteration has 2 stages)
        self.stage_num = None

        # labels ("x" variable in the paper), where "_nz" indicates that only the
        # nonmasked entries are stored.
        self.labels_nz = None

        # reflectance intensity (obtained from kmeans)
        self.intensities = None
        # reflectance chromaticity (obtained from kmeans)
        self.chromaticities = None

        # store here for visualization only
        self.shading_target = None

    def copy(self):
        ret = IntrinsicDecomposition(self.params, self.input)
        ret.iter_num = self.iter_num
        ret.stage_num = self.stage_num
        ret.labels_nz = self.labels_nz.copy()
        ret.intensities = self.intensities.copy()
        ret.chromaticities = self.chromaticities.copy()
        if self.shading_target is not None:
            ret.shading_target = self.shading_target.copy()
        return ret

    def get_r_s_nz(self):
        """ Return (reflectance, shading), with just the nonmasked entries """
        s_nz = self.input.image_gray_nz / self.intensities[self.labels_nz]
        r_nz = self.input.image_rgb_nz / np.clip(s_nz, 1e-4, 1e5)[:, np.newaxis]
        assert s_nz.ndim == 1 and r_nz.ndim == 2 and r_nz.shape[1] == 3
        return r_nz, s_nz

    def get_r_s(self):
        """ Return (reflectance, shading), in the full (rows, cols) shape """
        r_nz, s_nz = self.get_r_s_nz()
        r = np.zeros((self.input.rows, self.input.cols, 3), dtype=r_nz.dtype)
        s = np.zeros((self.input.rows, self.input.cols), dtype=s_nz.dtype)
        r[self.input.mask_nz] = r_nz
        s[self.input.mask_nz] = s_nz
        assert s.ndim == 2 and r.ndim == 3 and r.shape[2] == 3
        return r, s

    def get_r_gray(self):
        r_nz = self.intensities[self.labels_nz]
        r = np.zeros((self.input.rows, self.input.cols), dtype=r_nz.dtype)
        r[self.input.mask_nz] = r_nz
        return r

    def get_labels_visualization(self):
        #colors = image_util.n_distinct_colors(self.nlabels + 1)
        colors = self.get_reflectances_rgb()
        colors = np.vstack((colors, [0.0, 0.0, 0.0]))
        labels = self.get_labels()
        labels[labels == -1] = self.nlabels
        v = colors[labels, :]
        return v

    def get_reflectances_rgb(self):
        nlabels = self.intensities.shape[0]
        rgb = np.zeros((nlabels, 3))
        s = 3.0 * self.intensities
        r = self.chromaticities[:, 0]
        g = self.chromaticities[:, 1]
        b = 1.0 - r - g
        rgb[:, 0] = s * r
        rgb[:, 1] = s * g
        rgb[:, 2] = s * b
        return rgb

    def visualize_html(self):
        raise NotImplementedError("TODO")

    @property
    def nlabels(self):
        return self.intensities.shape[0]

    @property
    def input(self):
        return self._input

    def get_labels(self):
        """ Returns labels, expanded to the full image shape, with masked
        entries having a label of -1 """
        labels = np.empty((self.input.rows, self.input.cols), dtype=np.int32)
        labels.fill(-1)
        labels[self.input.mask_nz] = self.labels_nz
        return labels

    def save(self, solver, out_dir, save_extra=False, id=None):
        """ Save results to a directory """
        if not id:
            id = self.input.id
        if not id:
            raise ValueError("Need an id for saving")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        basename = os.path.join(out_dir, str(self.input.id))

        r, s = self.get_r_s()
        r_filename = '%s-r.png' % basename
        image_util.save(r_filename, r, mask_nz=solver.input.mask_nz,
                        rescale=True)

        s_filename = '%s-s.png' % basename
        image_util.save(s_filename, s, mask_nz=solver.input.mask_nz,
                        rescale=True)

        if save_extra:
            r_gray_filename = '%s-r-gray.png' % basename
            r_gray = self.get_r_gray()
            image_util.save(r_gray_filename, r_gray,
                            mask_nz=solver.input.mask_nz, rescale=True)

            labels_filename = '%s-labels.png' % basename
            labels_image = self.get_labels_visualization()
            image_util.save(labels_filename, labels_image,
                            mask_nz=solver.input.mask_nz, rescale=True)
