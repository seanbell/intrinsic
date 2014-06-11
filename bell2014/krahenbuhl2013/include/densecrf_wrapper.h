#include "densecrf.h"

class DenseCRFWrapper {
	public:
		DenseCRFWrapper(int npixels, int nlabels);
		virtual ~DenseCRFWrapper();

		void set_unary_energy(float* unary_costs_ptr);

		void add_pairwise_energy(float* pairwise_costs_ptr,
				float* features_ptr, int nfeatures);

		void map(int n_iters, int* result);

		int npixels();
		int nlabels();

	private:
		DenseCRF* m_crf;
		int m_npixels;
		int m_nlabels;
};
