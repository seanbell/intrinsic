import os
import sys
import csv
import gzip
import cPickle
import numpy as np

from ..density import ProbDensityHistogram


class ProbAbsoluteReflectance(object):
    """ Implements the absolute reflectance term p(R_x).  """

    def __init__(self, params):
        self.params = params
        self._load()

    def cost(self, r_nz):
        if self.params.abs_reflectance_weight:
            return self.params.abs_reflectance_weight * \
                (-self.density.logprob(np.log(r_nz)))
        else:
            return 0

    def _load(self):
        if self.params.logging:
            print("loading reflectances...")

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        data_filename = os.path.join(cur_dir, 'prob_abs_r.dat')
        if not os.path.exists(data_filename):
            rows = []
            to_id = {}
            with open(os.path.join(cur_dir, 'bsdfs.csv'), 'rb') as csvfile:
                first = True
                for row in csv.reader(csvfile):
                    if first:
                        to_id = {name: i for i, name in enumerate(row)}
                        first = False
                    else:
                        if row[to_id['colored_reflection']] == 'False':
                            r = float(row[to_id['rho_d_r']])
                            g = float(row[to_id['rho_d_g']])
                            b = float(row[to_id['rho_d_b']])
                            if r > 1e-4 and g > 1e-4 and b > 1e-4:
                                rows.append([r, g, b])
            data_raw = np.array(rows)

            data = np.clip(np.log(data_raw), np.log(1e-4), 0)
            self.density = ProbDensityHistogram()
            self.density.train(data, bins=100, bandwidth=3)

            cPickle.dump(
                obj=self.density,
                file=gzip.open(data_filename, "wb"),
                protocol=cPickle.HIGHEST_PROTOCOL
            )
        else:
            # make sure that 'bell2014.density' is on the path so that it
            # unpickles correclty
            path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if path not in sys.path:
                sys.path.append(path)

            self.density = cPickle.load(gzip.open(data_filename, "rb"))

        if self.params.logging:
            print("loaded reflectances")
