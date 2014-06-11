import json
import numpy as np


class HumanReflectanceJudgements(object):

    def __init__(self, judgements):
        if not isinstance(judgements, dict):
            raise ValueError("Invalid judgements: %s" % judgements)

        self.judgements = judgements
        self.id_to_points = {p['id']: p for p in self.points}

    @staticmethod
    def from_file(filename):
        judgements = json.load(open(filename))
        return HumanReflectanceJudgements(judgements)

    @property
    def points(self):
        return self.judgements['intrinsic_points']

    @property
    def comparisons(self):
        return self.judgements['intrinsic_comparisons']

    def compute_whdr(self, r, delta=0.10):
        """ Compute the Weighted Human Disagreement for a reflectance image
        ``r``.  """

        error_sum = 0.0
        weight_sum = 0.0

        for c in self.comparisons:
            point1 = self.id_to_points[c['point1']]
            point2 = self.id_to_points[c['point2']]
            darker = c['darker']
            weight = c['darker_score']

            if not point1['opaque'] or not point2['opaque']:
                continue
            if weight < 0 or weight is None:
                raise ValueError("Invalid darker_score: %s" % weight)
            if darker not in ('1', '2', 'E'):
                raise ValueError("Invalid darker: %s" % darker)

            l1 = np.mean(r[
                int(point1['y'] * r.shape[0]),
                int(point1['x'] * r.shape[1]),
                ...])
            l2 = np.mean(r[
                int(point2['y'] * r.shape[0]),
                int(point2['x'] * r.shape[1]),
                ...])

            l1 = max(l1, 1e-10)
            l2 = max(l2, 1e-10)

            if l2 / l1 > 1.0 + delta:
                alg_darker = '1'
            elif l1 / l2 > 1.0 + delta:
                alg_darker = '2'
            else:
                alg_darker = 'E'

            if darker != alg_darker:
                error_sum += weight
            weight_sum += weight

        if weight_sum:
            return error_sum / weight_sum
        else:
            return None
