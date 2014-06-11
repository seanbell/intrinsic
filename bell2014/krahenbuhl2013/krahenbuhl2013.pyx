# distutils: sources = src/densecrf_wrapper.cpp

cimport numpy as np

cdef extern from "include/densecrf_wrapper.h":
    cdef cppclass DenseCRFWrapper:
        DenseCRFWrapper(int, int) except +
        void set_unary_energy(float*)
        void add_pairwise_energy(float*, float*, int)
        void map(int, int*)
        int npixels()
        int nlabels()

cdef class DenseCRF:
    cdef DenseCRFWrapper *thisptr

    def __cinit__(self, int npixels, int nlabels):
        self.thisptr = new DenseCRFWrapper(npixels, nlabels)

    def __dealloc__(self):
        del self.thisptr

    def set_unary_energy(self, float[:, ::1] unary_costs):
        if (unary_costs.shape[0] != self.thisptr.npixels() or
                unary_costs.shape[1] != self.thisptr.nlabels()):
            raise ValueError("Invalid unary_costs shape")

        self.thisptr.set_unary_energy(&unary_costs[0, 0])

    def add_pairwise_energy(self, float[:, ::1] pairwise_costs,
                            float[:, ::1] features):
        if (pairwise_costs.shape[0] != self.thisptr.nlabels() or
                pairwise_costs.shape[1] != self.thisptr.nlabels()):
            raise ValueError("Invalid pairwise_costs shape")
        if (features.shape[0] != self.thisptr.npixels()):
            raise ValueError("Invalid features shape")

        self.thisptr.add_pairwise_energy(
            &pairwise_costs[0, 0],
            &features[0, 0],
            features.shape[1]
        )

    def map(self, int n_iters=10):
        import numpy as np
        labels = np.empty(self.thisptr.npixels(), dtype=np.int32)
        cdef int[::1] labels_view = labels
        self.thisptr.map(n_iters, &labels_view[0])
        return labels
