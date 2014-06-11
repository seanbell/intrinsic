import timeit
import numpy as np

from ..image_util import gaussian_blur_gray_image_nz
from .prob_abs_r import ProbAbsoluteReflectance
from .prob_abs_s import ProbAbsoluteShading


class IntrinsicEnergy(object):

    def __init__(self, input, params):
        self.input = input
        self.params = params
        self.prob_abs_r = ProbAbsoluteReflectance(params)
        self.prob_abs_s = ProbAbsoluteShading(params)

    def compute_unary_costs(self, decomposition, prev_decomposition):
        """ Returns unary costs: nnz x nlabels matrix """

        if self.params.logging:
            t0 = timeit.default_timer()
            print("compute_unary_costs...")

        intensities = decomposition.intensities
        chromaticities = decomposition.chromaticities
        nlabels = intensities.shape[0]
        unary_costs = np.zeros(
            (self.input.mask_nnz, nlabels),
            dtype=np.float32)

        sigma_spatial = (
            self.params.shading_blur_sigma *
            self.input.diag / (
                1.0 + decomposition.iter_num **
                self.params.shading_blur_iteration_pow
            )
        )
        if self.params.logging:
            print('blur sigma: %s pixels (image diagonal: %s pixels)' %
                  (sigma_spatial, self.input.diag))

        # obtain previous shading layer, or use a method to create a proxy
        if prev_decomposition:
            prev_r_nz, prev_s_nz = prev_decomposition.get_r_s_nz()
        elif self.params.shading_blur_init_method == "constant":
            prev_s_nz = 0.5 * np.ones_like(self.input.image_gray_nz)
        elif self.params.shading_blur_init_method == "image":
            prev_s_nz = self.input.image_gray_nz
        elif self.params.shading_blur_init_method == "none":
            prev_s_nz = None
        else:
            raise ValueError("Unknown shading_blur_init_method: %s" %
                             self.params.shading_blur_init_method)

        if prev_s_nz is not None:
            if self.params.shading_blur_log:
                # blur in log space
                blur_input = np.log(prev_s_nz)
            else:
                # blur in linear space, then convert to log
                blur_input = prev_s_nz

            blur_output = gaussian_blur_gray_image_nz(
                image_nz=blur_input,
                image_shape=self.input.shape,
                mask_nz=self.input.mask_nz,
                sigma=sigma_spatial,
            )

            if self.params.shading_blur_log:
                log_s_target_nz = blur_output
            else:
                log_s_target_nz = np.log(blur_output)
        else:
            log_s_target_nz = None

        # (used below)
        if self.params.shading_target_chromaticity:
            labels_rgb = np.clip(
                decomposition.get_reflectances_rgb(), 1e-5, np.inf)

        # shading and chromaticity terms
        for i in xrange(nlabels):
            s_nz = self.input.image_gray_nz / intensities[i]
            r_nz = (self.input.image_rgb_nz /
                    np.clip(s_nz, 1e-4, 1e5)[:, np.newaxis])

            # absolute reflectance and shading
            unary_costs[:, i] += (
                self.prob_abs_s.cost(s_nz) +
                self.prob_abs_r.cost(r_nz)
            )

            # chromaticity: encourage reflectance intensities to be assigned to
            # pixels that share the same chromaticity as the original kmeans
            # cluster from which the reflectance intensity was obtained.
            if self.params.chromaticity_weight:
                if self.params.chromaticity_norm == "L1":
                    f = np.abs
                elif self.params.chromaticity_norm == "L2":
                    f = np.square
                else:
                    raise ValueError(
                        "Invalid value of chromaticity_norm: %s" %
                        self.params.chromaticity_norm)

                unary_costs[:, i] += self.params.chromaticity_weight * (
                    np.sum(
                        f(self.input.image_irg_nz[:, 1:3] -
                          chromaticities[i, :]),
                        axis=1
                    )
                )

            # shading smoothness: discourage shading discontinuities
            if self.params.shading_target_weight and log_s_target_nz is not None:
                if self.params.shading_target_norm == "L2":
                    f = np.square
                elif self.params.shading_target_norm == "L1":
                    f = np.abs
                else:
                    raise ValueError("Invalid value of shading_target_norm: %s" %
                                     self.params.shading_target_norm)

                if self.params.shading_target_chromaticity:
                    # interpret labels as RGB (intensity with chromaticity),
                    # thereby penalizing deviations from grayscale in the
                    # shading channel (though the final answer is always
                    # grayscale anyway)
                    label_rgb = labels_rgb[i, :]
                    s_rgb_nz = self.input.image_rgb_nz / label_rgb[np.newaxis, :]
                    log_s_rgb_nz = np.log(np.clip(s_rgb_nz, 1e-5, np.inf))
                    unary_costs[:, i] += (
                        self.params.shading_target_weight *
                        np.sum(f(log_s_rgb_nz - log_s_target_nz[:, np.newaxis]), axis=-1)
                    )
                else:
                    # interpret labels as intensities
                    log_s_nz = np.log(s_nz)
                    unary_costs[:, i] += (
                        self.params.shading_target_weight *
                        f(log_s_nz - log_s_target_nz)
                    )

        if self.params.logging:
            t1 = timeit.default_timer()
            print("compute_unary_costs: done (%s s)" % (t1 - t0))

        return unary_costs

    def compute_pairwise_costs(self, decomposition):
        """ Returns the pairwise cost matrix: nlabels x nlabels matrix.
        Entry ij is ``abs(intensity[i] - intensity[j])`` """

        if self.params.pairwise_intensity_chromaticity:
            # interpret labels as RGB (intensity with chromaticity)
            nlabels = decomposition.intensities.shape[0]
            R = decomposition.get_reflectances_rgb()
            if self.params.pairwise_intensity_log:
                R = np.log(np.clip(R, 1e-5, np.inf))
            binary_costs = np.zeros((nlabels, nlabels), dtype=np.float32)
            for i in xrange(nlabels):
                for j in xrange(i):
                    cost = np.sum(np.abs(R[i, :] - R[j, :]))
                    binary_costs[i, j] = cost
                    binary_costs[j, i] = cost
        else:
            # interpret labels as intensities
            R = decomposition.intensities
            if self.params.pairwise_intensity_log:
                R = np.log(np.clip(R, 1e-5, np.inf))
            binary_costs = np.abs(R[:, np.newaxis] - R[np.newaxis, :])

        return binary_costs

    def get_features(self):
        """ Return an nnz x nfeatures matrix containing the features """

        if not hasattr(self, '_features'):
            mask_nz = self.input.mask_nz
            mask_nnz = self.input.mask_nnz
            features = np.zeros((mask_nnz, 5), dtype=np.float32)

            # image intensity
            features[:, 0] = (
                self.input.image_irg[mask_nz[0], mask_nz[1], 0] /
                self.params.theta_l)

            # image chromaticity
            features[:, 1] = (
                self.input.image_irg[mask_nz[0], mask_nz[1], 1] /
                self.params.theta_c)
            features[:, 2] = (
                self.input.image_irg[mask_nz[0], mask_nz[1], 2] /
                self.params.theta_c)

            # pixel location
            features[:, 3] = (
                mask_nz[0] / (self.params.theta_p * self.input.diag))
            features[:, 4] = (
                mask_nz[1] / (self.params.theta_p * self.input.diag))

            self._features = features
            self._features.setflags(write=False)

        return self._features
