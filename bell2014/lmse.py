"""
LMSE code from MIT Intrinsic Images dataset:

    Roger Grosse, Micah K. Johnson, Edward H. Adelson, and William T. Freeman,
    Ground truth dataset and baseline evaluations for intrinsic image
    algorithms, in Proceedings of the International Conference on Computer
    Vision (ICCV), 2009.
    http://people.csail.mit.edu/rgrosse/intrinsic/

NOTE: a small fix was included (see comment in local_error)

"""

import numpy as np


def ssq_error(correct, estimate, mask):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate**2 * mask) > 1e-5:
        alpha = np.sum(correct * estimate * mask) / np.sum(estimate**2 * mask)
    else:
        alpha = 0.
    return np.sum(mask * (correct - alpha*estimate) ** 2)


def local_error(correct, estimate, mask, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N = correct.shape[:2]
    ssq = total = 0.
    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):
            correct_curr = correct[i:i+window_size, j:j+window_size]
            estimate_curr = estimate[i:i+window_size, j:j+window_size]
            mask_curr = mask[i:i+window_size, j:j+window_size]
            ssq += ssq_error(correct_curr, estimate_curr, mask_curr)
            # FIX: in the original codebase, this was outdented, which allows
            # for scores greater than 1 (which should not be possible).  On the
            # MIT dataset images, this makes a negligible difference, but on
            # larger images, this can have a significant effect.
            total += np.sum(mask_curr * correct_curr**2)
    assert -np.isnan(ssq/total)

    return ssq / total


def score_image(true_shading, true_refl, estimate_shading, estimate_refl, mask, window_size=20):
    return 0.5 * local_error(true_shading, estimate_shading, mask, window_size, window_size//2) + \
           0.5 * local_error(true_refl, estimate_refl, mask, window_size, window_size//2)
