import numpy as np
from scipy.ndimage.filters import gaussian_filter


class ProbDensityHistogram(object):

    def train(self, training_data, bins=100, bandwidth=3, smoothing=1e-2):
        self.ndim = training_data.shape[1]
        self.bins = bins
        self.hist, self.edges = np.histogramdd(
            training_data, bins=bins, normed=True)
        self.hist[self.hist < smoothing] = smoothing
        if bandwidth:
            self.hist = gaussian_filter(self.hist, sigma=bandwidth)
        #self.hist /= np.sum(self.hist)
        self.hist = np.log(self.hist)

    def logprob(self, samples):
        indices = [np.digitize(samples[:, i], self.edges[i])
                   for i in xrange(self.ndim)]
        for i in xrange(self.ndim):
            np.clip(indices[i], 0, self.bins - 1, out=indices[i])
        ret = self.hist[indices]
        assert ret.shape[0] == samples.shape[0]
        return ret
