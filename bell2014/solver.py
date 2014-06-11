import timeit
import numpy as np
import sklearn
from skimage import morphology
from sklearn.cluster import MiniBatchKMeans

from .params import IntrinsicParameters
from .decomposition import IntrinsicDecomposition
from .energy import IntrinsicEnergy
from .optimization import minimize_l1, minimize_l2

from .krahenbuhl2013.krahenbuhl2013 import DenseCRF


class IntrinsicSolver(object):

    def __init__(self, input, params):
        """ Create a new solver with given input and parameters.  Nothing
        happens until you call ``solve``. """

        if isinstance(params, dict):
            params = IntrinsicParameters.from_dict(params)

        self.params = params
        self.input = input
        self.energy = IntrinsicEnergy(self.input, params)

    def solve(self):
        """ Perform all steps. """

        if self.params.logging:
            t0 = timeit.default_timer()
            print("solve...")

        # Initialize
        self.decomposition = IntrinsicDecomposition(self.params, self.input)
        self.decomposition_history = []
        self.initialize_intensities()

        for i in xrange(self.params.n_iters):
            if self.params.logging:
                print("\nrun: starting iteration %s/%s" % (i, self.params.n_iters))
            self.decomposition.iter_num = i

            # STAGE 1
            self.decomposition.stage_num = 1
            self.stage1_optimize_r()
            self.remove_unused_intensities()
            self.decomposition_history.append(self.decomposition.copy())

            if self.decomposition.intensities.shape[0] <= 1:
                if self.params.logging:
                    print("Warning: only 1 reflectance -- exit early")
                break

            # STAGE 2
            self.decomposition.stage_num = 2
            if self.params.split_clusters and i == self.params.n_iters - 1:
                self.split_label_clusters()
            self.stage2_smooth_s()
            self.decomposition_history.append(self.decomposition.copy())

        # prepare final solution
        r, s = self.decomposition.get_r_s()

        if self.params.logging:
            t1 = timeit.default_timer()
            print("solve (%s s)" % (t1 - t0))

        return r, s, self.decomposition

    def prev_decomposition(self):
        """ Return the previous decomposition (used to compute the blurred
        shading target). """

        if self.decomposition_history:
            return self.decomposition_history[-1]
        else:
            return None

    def initialize_intensities(self):
        """ Initialization: k-means of the input image """

        if self.params.logging:
            t0 = timeit.default_timer()
            print("initialization: k-means clustering with %s centers..." %
                  self.params.kmeans_n_clusters)

        image_irg = self.input.image_irg
        mask_nz = self.input.mask_nz

        if self.params.fixed_seed:
            # fix the seed when computing things like gradients across
            # hyperparameters
            random_state = np.random.RandomState(seed=59173)
        else:
            random_state = None

        samples = image_irg[mask_nz[0], mask_nz[1], :]
        if samples.shape[0] > self.params.kmeans_max_samples:
            print("image is large: subsampling %s/%s random pixels" %
                  (self.params.kmeans_max_samples, samples.shape[0]))
            samples = sklearn.utils \
                .shuffle(samples)[:self.params.kmeans_max_samples, :]
        samples[:, 0] *= self.params.kmeans_intensity_scale

        kmeans = MiniBatchKMeans(
            n_clusters=self.params.kmeans_n_clusters,
            compute_labels=False, random_state=random_state)
        kmeans.fit(samples)

        assert self.params.kmeans_intensity_scale > 0
        self.decomposition.intensities = (
            kmeans.cluster_centers_[:, 0] /
            self.params.kmeans_intensity_scale
        )
        self.decomposition.chromaticities = (
            kmeans.cluster_centers_[:, 1:3]
        )

        if self.params.logging:
            t1 = timeit.default_timer()
            print("clustering done (%s s).  intensities:\n%s" %
                  (t1 - t0, self.decomposition.intensities))

    def stage1_optimize_r(self):
        """ Stage 1: dense CRF optimization """

        if self.params.logging:
            t0 = timeit.default_timer()
            print("stage1_optimize_r: compute costs...")

        nlabels = self.decomposition.intensities.shape[0]
        npixels = self.input.mask_nnz

        # use a Python wrapper around the code from [Krahenbuhl et al 2013]
        densecrf = DenseCRF(npixels, nlabels)

        # unary costs
        unary_costs = self.energy.compute_unary_costs(
            decomposition=self.decomposition,
            prev_decomposition=self.prev_decomposition(),
        )
        densecrf.set_unary_energy(unary_costs)

        # pairwise costs
        if self.params.pairwise_weight:
            pairwise_costs = self.energy.compute_pairwise_costs(
                decomposition=self.decomposition,
            )
            densecrf.add_pairwise_energy(
                pairwise_costs=(self.params.pairwise_weight * pairwise_costs).astype(np.float32),
                features=self.energy.get_features().copy(),
            )

        if self.params.logging:
            print("stage1_optimize_r: optimizing dense crf (%s iters)..." %
                  self.params.n_crf_iters)
            t0crf = timeit.default_timer()

        # maximum aposteriori labeling ("x" variable in the paper)
        self.decomposition.labels_nz = densecrf.map(self.params.n_crf_iters)

        if self.params.logging:
            t1crf = timeit.default_timer()
            print("stage1_optimize_r: dense crf done (%s s)" % (t1crf - t0crf))

        if self.params.logging:
            t1 = timeit.default_timer()
            print("stage1_optimize_r: done (%s s)" % (t1 - t0))

    def stage2_smooth_s(self):
        """ Stage 2: L1 shading smoothness """

        if self.params.logging:
            t0 = timeit.default_timer()
            print('stage2_smooth_s: constructing linear system...')

        if self.params.stage2_maintain_median_intensity:
            median_intensity = np.median(self.decomposition.intensities)

        log_intensities = np.log(self.decomposition.intensities)

        # the 'A' matrix (in Ax = b) is in CSR sparse format
        A_data, A_rows, A_cols, A_shape, b = \
            self.construct_shading_smoothness_system(log_intensities)

        if len(b) < 1:
            if self.params.logging:
                print('Warning: empty linear system (%s, nlabels=%s)' % (
                    self.basename, self.cur_intensities.shape[0]))
            return

        if self.params.logging:
            print('solving linear system...')

        # solve for the change to the variables, so that we can slightly
        # regularize the variables to be near zero (i.e. near the previous
        # value).
        if self.params.stage2_norm == "L1":
            minimize = minimize_l1
        elif self.params.stage2_norm == "L2":
            minimize = minimize_l2
        else:
            raise ValueError("Invalid stage2_norm: %s" % self.params.stage2_norm)
        delta_intensities = minimize(
            A_data, A_rows, A_cols, A_shape, b,
            damp=1e-8, logging=self.params.logging,
        )
        intensities = np.exp(log_intensities + delta_intensities)

        if self.params.stage2_maintain_median_intensity:
            # Since there's a scale ambiguity and stage1 includes a term that
            # depends on absolute shading, keep the median intensity constant.
            # This is a pretty small adjustment.
            intensities *= median_intensity / np.median(intensities)

        self.decomposition.intensities = intensities

        if self.params.logging:
            t1 = timeit.default_timer()
            print('stage2_smooth_s: done (%s s)' % (t1 - t0))

    def construct_shading_smoothness_system(self, log_intensities):
        """ Create a sparse matrix (CSR format) to minimize discontinuities in
        the shading channel (by adjusting ``decomposition.intensities``).

        :return: A_data, A_rows, A_cols, A_shape, b
        """

        rows, cols = self.input.shape[0:2]
        mask = self.input.mask
        labels = self.decomposition.get_labels()

        if self.params.stage2_chromaticity:
            # labels represent RGB colors (but we are still only adjusting
            # intensity)
            log_image_rgb = self.input.log_image_rgb
            log_reflectances_rgb = np.log(np.clip(self.decomposition.get_reflectances_rgb(), 1e-5, np.inf))
        else:
            # labels represent intensities
            log_image_gray = self.input.log_image_gray

        A_rows = []
        A_cols = []
        A_data = []
        b = []

        # Note that in the paper, params.shading_smooth_k = 1, i.e.  only the
        # immediate pixel neighbors are smoothed.  This code is slightly more
        # general in that it allows to smooth pixels k units away if you set
        # shading_smooth_k > 1, weighted by 1/(k*k).
        for k in xrange(1, self.params.shading_smooth_k + 1):
            weight = 1.0 / (k * k)
            for i in xrange(rows - k):
                for j in xrange(cols - k):
                    if not mask[i, j]:
                        continue
                    if mask[i + k, j]:
                        l0 = labels[i, j]
                        l1 = labels[i + k, j]
                        if l0 != l1:
                            if self.params.stage2_chromaticity:
                                # RGB interpretation
                                for c in xrange(3):
                                    A_rows.append(len(b))
                                    A_cols.append(l0)
                                    A_data.append(weight)
                                    A_rows.append(len(b))
                                    A_cols.append(l1)
                                    A_data.append(-weight)
                                    bval = (log_image_rgb[i, j, c] -
                                            log_image_rgb[i + k, j, c] +
                                            log_reflectances_rgb[l1, c] -
                                            log_reflectances_rgb[l0, c])
                                    b.append(weight * bval)
                            else:
                                # intensity interpretation
                                A_rows.append(len(b))
                                A_cols.append(l0)
                                A_data.append(weight)
                                A_rows.append(len(b))
                                A_cols.append(l1)
                                A_data.append(-weight)
                                bval = (log_image_gray[i, j] -
                                        log_image_gray[i + k, j] +
                                        log_intensities[l1] -
                                        log_intensities[l0])
                                b.append(weight * bval)
                    if mask[i, j + k]:
                        l0 = labels[i, j]
                        l1 = labels[i, j + k]
                        if l0 != l1:
                            if self.params.stage2_chromaticity:
                                # RGB interpretation
                                for c in xrange(3):
                                    A_rows.append(len(b))
                                    A_cols.append(l0)
                                    A_data.append(weight)
                                    A_rows.append(len(b))
                                    A_cols.append(l1)
                                    A_data.append(-weight)
                                    bval = (log_image_rgb[i, j, c] -
                                            log_image_rgb[i, j + k, c] +
                                            log_reflectances_rgb[l1, c] -
                                            log_reflectances_rgb[l0, c])
                                    b.append(weight * bval)
                            else:
                                # intensity interpretation
                                A_rows.append(len(b))
                                A_cols.append(l0)
                                A_data.append(weight)
                                A_rows.append(len(b))
                                A_cols.append(l1)
                                A_data.append(-weight)
                                bval = (log_image_gray[i, j] -
                                        log_image_gray[i, j + k] +
                                        log_intensities[l1] -
                                        log_intensities[l0])
                                b.append(weight * bval)

        A_shape = (len(b), log_intensities.shape[0])
        return (
            np.array(A_data),
            np.array(A_rows),
            np.array(A_cols),
            A_shape,
            np.array(b, dtype=np.float)
        )

    def remove_unused_intensities(self):
        """ Remove any intensities that are not currently assigned to a pixel,
        and then re-number all labels so they are contiguous again. """

        if self.params.logging:
            prev_r_s = self.decomposition.get_r_s()

        labels_nz = self.decomposition.labels_nz
        intensities = self.decomposition.intensities
        chromaticities = self.decomposition.chromaticities
        nlabels = intensities.shape[0]

        new_to_old = np.nonzero(np.bincount(
            labels_nz, minlength=nlabels))[0]
        old_to_new = np.empty(nlabels, dtype=np.int32)
        old_to_new.fill(-1)
        for new, old in enumerate(new_to_old):
            old_to_new[old] = new

        self.decomposition.labels_nz = old_to_new[labels_nz]
        self.decomposition.intensities = intensities[new_to_old]
        self.decomposition.chromaticities = chromaticities[new_to_old]

        if self.params.logging:
            print ('remove_unused_intensities: %s/%s labels kept' % (
                   len(self.decomposition.intensities), len(intensities)))

        if self.params.logging:
            np.testing.assert_equal(self.decomposition.get_r_s(), prev_r_s)
            assert (self.decomposition.chromaticities.shape[0] ==
                    self.decomposition.intensities.shape[0])

    def split_label_clusters(self, neighbors=4):
        """ Expand the set of labels by looking at each connected component in
        the labels.  Assign each component a new label number, and copy its old
        intensity value to its new label. This typically expands the number of
        labels from ~30 to ~3000, so you should only really do it on the last
        iteration. """

        if self.params.logging:
            prev_r_s = self.decomposition.get_r_s()

        rows, cols = self.input.shape[0:2]
        labels = self.decomposition.get_labels()
        intensities = self.decomposition.intensities
        chromaticities = self.decomposition.chromaticities

        # split labels
        new_labels = morphology.label(labels, neighbors=neighbors)

        # map labels
        self.decomposition.labels_nz = new_labels[self.input.mask_nz]

        # map intensities
        _, indices = np.unique(new_labels.ravel(), return_index=True)
        new_to_old = labels.ravel()[indices]
        new_to_old = new_to_old[new_to_old != -1]
        self.decomposition.intensities = intensities[new_to_old]
        self.decomposition.chromaticities = chromaticities[new_to_old]

        if self.params.logging:
            print ('split_label_clusters: %s --> %s' % (
                   intensities.shape[0], self.decomposition.intensities.shape[0]))

        self.remove_unused_intensities()

        if self.params.logging:
            np.testing.assert_equal(self.decomposition.get_r_s(), prev_r_s)
            assert (self.decomposition.chromaticities.shape[0] ==
                    self.decomposition.intensities.shape[0])
