import scipy
import numpy as np


def minimize_l2(A_data, A_rows, A_cols, A_shape, b, damp=1e-8, logging=False):
    A = scipy.sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=A_shape)
    return scipy.sparse.linalg.lsmr(A, b, damp=damp)[0]


def minimize_l1(A_data, A_rows, A_cols, A_shape, b, x0=None,
                tol=1e-6, irls_epsilon=1e-6, damp=1e-8,
                max_iters=100, logging=False):
    """
    Perform L1 minimization of ``sum(|A.dot(x) - b|)`` via iteratively
    reweighted least squares.
    """

    if logging:
        print('solving sparse linear system (%s x %s, %s nnz)...' % (
              A_shape[0], A_shape[1], len(A_data)))
    if A_shape[0] == 0 or A_shape[1] == 0 or b.shape[0] == 0:
        print('Warning: empty linear system! returning 0')
        return np.zeros(A_shape[1])

    # construct matrix
    A = scipy.sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=A_shape, dtype=np.float)

    # initial solution
    if x0 is not None:
        x = x0
    else:
        x = scipy.sparse.linalg.lsmr(A, b, damp=damp)[0]

    prev_x = x
    prev_mean_error = float('inf')

    for i in xrange(max_iters):
        error = np.abs(A.dot(x) - b)
        mean_error = np.mean(error)

        if logging and i % 10 == 0:
            print('l1 optimization: (iter %s) mean_error: %s' % (i, mean_error))

        # exit conditions
        delta_error = prev_mean_error - mean_error
        if delta_error < 0:
            if logging:
                print('l1 optimization: (iter %s) mean_error increased: %s --> %s (exit)' %
                      (i, prev_mean_error, mean_error))
            return prev_x
        elif delta_error < tol:
            if logging:
                print('l1 optimization: (iter %s) mean_error: %s, delta_error: %s < %s (exit)' %
                      (i, mean_error, delta_error, tol))
            return x

        prev_x = x
        prev_mean_error = mean_error

        # solve next problem
        w = np.sqrt(np.reciprocal(error + irls_epsilon))
        Aw_data = A_data * w[A_rows]
        Aw = scipy.sparse.csr_matrix((Aw_data, (A_rows, A_cols)), shape=A_shape)
        bw = b * w
        x = scipy.sparse.linalg.lsmr(Aw, bw, damp=damp)[0]

    if logging:
        print ('l1 optimization: did not converge within %s iterations' %
               max_iters)
    return x
