import numpy as np


class ProbAbsoluteShading(object):
    def __init__(self, params):
        self.params = params

    def cost(self, s_nz):
        if self.params.abs_shading_weight:
            if self.params.abs_shading_log:
                return self.params.abs_shading_weight * \
                    np.abs(np.log(s_nz) - np.log(self.params.abs_shading_gray_point))
            else:
                return self.params.abs_shading_weight * \
                    np.abs(s_nz - self.params.abs_shading_gray_point)
        else:
            return 0
