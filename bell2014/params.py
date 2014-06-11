import copy
import json
import random
import hashlib
import numpy as np


class IntrinsicParameters():
    """ Global parameter values for the algorithm """

    def __init__(self):

        #: if True, print progress to the console
        self.logging = False

        #: if True, use a fixed seed for k-means clustering
        self.fixed_seed = False

        #: number of iterations for the global loop
        self.n_iters = 25

        #: number of iterations for the dense CRF
        self.n_crf_iters = 10

        #: if ``True``, split clusters at the end
        self.split_clusters = True

        #: Pixels k units apart vertically or horizontally are smoothed.
        #: The paper only uses k=1.
        self.shading_smooth_k = 1

        #: method used to initialize the shading smoothness term:
        #:   "none": omit this term for the first iteration
        #:   "image": use the image itself (intensity channel)
        #:   "constant": constant 0.5
        self.shading_blur_init_method = 'none'

        #: standard deviation for blurring the shading channel
        self.shading_blur_sigma = 0.1

        #: exponent by which the blur size decreases each iteration
        self.shading_blur_iteration_pow = 1

        #: if ``True``, blur in log space.  if ``False``, blur in linear
        #: space and then convert to log.
        self.shading_blur_log = True

        #: kmeans initialization: weight given to the intensity channel
        self.kmeans_intensity_scale = 0.5

        #: kmeans initialization: number of clusters (labels) to use
        self.kmeans_n_clusters = 20

        #: kmeans initialization: max pixels to consider at once
        #: (if the image has more than this, the image is randomly subsampled)
        self.kmeans_max_samples = 2000000

        #: weight of the absolute reflectance prior
        self.abs_reflectance_weight = 0

        #: weight of the absolute shading prior
        self.abs_shading_weight = 500.0

        #: gray-point of absolute shading term
        self.abs_shading_gray_point = 0.5

        #: if ``True``, compute shading error in log space
        self.abs_shading_log = True

        #: weight of the shading smoothness unary term
        self.shading_target_weight = 20000.0

        #: norm used to penalize shading smoothness deviations
        self.shading_target_norm = "L2"

        #: interpret labels as RGB (intensity with chromaticity), thereby
        #: penalizing deviations from grayscale in the shading channel (though
        #: the final answer is always grayscale anyway)
        self.shading_target_chromaticity = False

        #: weight of the chromaticity term: each reflectance intensity is
        #: assigned a chromaticity (from the kmeans initialization) and is
        #: encouraged to be assigned to image pixels that share the same
        #: chromaticity.
        self.chromaticity_weight = 0

        #: which norm is used for chromaticity
        self.chromaticity_norm = "L1"

        #: compute reflectance distance in log space for the pairwise terms
        self.pairwise_intensity_log = True

        #: include chromaticity in pairwise term
        self.pairwise_intensity_chromaticity = True

        #: weight of the pairwise term
        self.pairwise_weight = 10000.0

        #: bilateral standard deviation: pairwise pixel distance
        self.theta_p = 0.1

        #: bilateral standard deviation: intensity
        self.theta_l = 0.1

        #: bilateral standard deviation: chromaticity
        self.theta_c = 0.025

        #: if True, keep the median of all intensities fixed in stage 2.  This
        #: doesn't really change much, since the solver is damped anyway.
        self.stage2_maintain_median_intensity = True

        #: which norm to use when minimizing shading differences in stage 2
        self.stage2_norm = "L1"

        #: if True, interpret labels as RGB instead of intensity
        self.stage2_chromaticity = False

    #: parameters to be saved/loaded
    ALL_PARAMS = [
        'n_iters',
        'n_crf_iters',
        'split_clusters',
        'kmeans_n_clusters',
        'kmeans_max_samples',
        'shading_blur_init_method',
        'shading_blur_method',
        'shading_blur_log',
        'shading_blur_sigma',
        'shading_blur_bilateral_sigma_range',
        'shading_blur_iteration_pow',
        'shading_smooth_k',
        'kmeans_intensity_scale',
        'abs_reflectance_weight',
        'abs_shading_log',
        'abs_shading_weight',
        'abs_shading_gray_point',
        'shading_target_weight',
        'shading_target_norm',
        'shading_target_chromaticity',
        'chromaticity_weight',
        'chromaticity_norm',
        'pairwise_intensity_log',
        'pairwise_intensity_chromaticity',
        'pairwise_weight',
        'theta_p',
        'theta_l',
        'theta_c',
        'stage2_norm',
        'stage2_chromaticity',
        'stage2_maintain_median_intensity',
    ]

    #: parameters to be adjusted during training
    TRAIN_PARAMS = [
        'n_iters',
        #'n_crf_iters',

        'split_clusters',

        'kmeans_intensity_scale',
        'kmeans_n_clusters',

        'shading_blur_init_method',
        #'shading_blur_log',
        #'pairwise_intensity_log',

        'shading_blur_sigma',
        'shading_smooth_k',

        'abs_reflectance_weight',
        #'abs_shading_log',
        'abs_shading_weight',
        'abs_shading_gray_point',
        'shading_target_weight',
        'chromaticity_weight',
        'pairwise_weight',

        'theta_p',
        'theta_l',
        'theta_c',
    ]

    #: these parameters are discrete 1-of-N choices
    PARAM_CHOICES = {
        'shading_blur_init_method': (
            "none",
            "image",
            "constant",
        ),
    }

    #: bounds on paramters
    PARAM_BOUNDS = {
        'n_iters': (1, 30),
        'n_crf_iters': (1, 10),
        'shading_blur_sigma': (1e-8, 1.0),
        'shading_smooth_k': (1, 4),
        'kmeans_intensity_scale': (1e-8, 1e10),
        'kmeans_n_clusters': (2, 50),
        'abs_reflectance_weight': (0, 1e10),
        'abs_shading_weight': (0, 1e10),
        'abs_shading_gray_point': (0, 1e10),
        'shading_target_weight': (0, 1e10),
        'chromaticity_weight': (0, 1e10),
        'pairwise_weight': (0, 1e16),
        'theta_p': (1e-8, 1e10),
        'theta_l': (1e-8, 1e10),
        'theta_c': (1e-8, 1e10),
    }

    WEIGHT_PARAMS = [
        'abs_reflectance_weight',
        'abs_shading_weight',
        'shading_target_weight',
        'chromaticity_weight',
        'pairwise_weight',
    ]

    THETA_PARAMS = [
        'theta_p',
        'theta_l',
        'theta_c',
    ]

    def to_json(self, indent=4, **extra_kwargs):
        """ Convert paramters to a JSON-encoded string """
        obj = {k: getattr(self, k)
               for k in IntrinsicParameters.ALL_PARAMS}
        if extra_kwargs:
            obj.update(extra_kwargs)
        return json.dumps(obj, sort_keys=True, indent=indent)

    def __str__(self):
        return self.to_json()

    def __unicode__(self):
        return self.to_json()

    @staticmethod
    def from_file(filename):
        """ Load paramers from ``filename`` (in JSON format) """
        return IntrinsicParameters.from_dict(json.load(open(filename)))

    @staticmethod
    def from_dict(d):
        """ Load paramers from a dictionary """
        ret = IntrinsicParameters()
        for k, v in d.iteritems():
            if not k.startswith('_') and k not in IntrinsicParameters.ALL_PARAMS:
                raise ValueError("Invalid parameter: %s" % k)
            setattr(ret, k, d[k])
        return ret

    def md5(self):
        dump = self.to_json()
        m = hashlib.md5()
        m.update(dump)
        return m.hexdigest()

    def save(self, filename, **extra_kwargs):
        """ Save paramers to ``filename`` (in JSON format) """
        with open(filename, 'w') as f:
            f.write(self.to_json(**extra_kwargs))

    def clip(self):
        """ Clip parameters to be within bounds """
        for k, bounds in IntrinsicParameters.PARAM_BOUNDS.iteritems():
            v = getattr(self, k)
            t = type(v)
            setattr(self, k, t(np.clip(v, bounds[0], bounds[1])))

    def random_perterbation(
            self, mean_num_params=8, std_delta=0.5, seed=None):
        """ Return a new set of parameters with a random perterbation.  The
        number of variables modified is Poisson-distributed with mean
        ``mean_num_params`` , and each changed variable is multiplied by exp(x)
        where x is normally distributed with mean 0 and standard deviation
        ``std_delta`` """

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # choose a random subset to modify
        num_params = len(IntrinsicParameters.TRAIN_PARAMS)
        n = np.clip(np.random.poisson(mean_num_params), 1, num_params)
        keys = random.sample(IntrinsicParameters.TRAIN_PARAMS, n)

        # modify the subset
        ret = copy.deepcopy(self)
        for k in keys:
            v = getattr(ret, k)
            t = type(v)

            if k in IntrinsicParameters.PARAM_CHOICES:
                v = random.choice(IntrinsicParameters.PARAM_CHOICES[k])
            elif t == bool:
                v = random.choice((False, True))
            else:
                v *= np.exp(random.normalvariate(0, std_delta))

            if t in (int, long):
                v = round(v)
            setattr(ret, k, t(v))

        ret.clip()
        return ret
