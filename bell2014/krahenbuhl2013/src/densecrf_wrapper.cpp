#include <Eigen/Core>
#include "densecrf.h"
#include "densecrf_wrapper.h"

DenseCRFWrapper::DenseCRFWrapper(int npixels, int nlabels)
: m_npixels(npixels), m_nlabels(nlabels) {
	m_crf = new DenseCRF(npixels, nlabels);
}

DenseCRFWrapper::~DenseCRFWrapper() {
	delete m_crf;
}

int DenseCRFWrapper::npixels() { return m_npixels; }
int DenseCRFWrapper::nlabels() { return m_nlabels; }

void DenseCRFWrapper::add_pairwise_energy(float* pairwise_costs_ptr, float* features_ptr, int nfeatures) {
	m_crf->addPairwiseEnergy(
		Eigen::Map<const Eigen::MatrixXf>(features_ptr, nfeatures, m_npixels),
		new MatrixCompatibility(
			Eigen::Map<const Eigen::MatrixXf>(pairwise_costs_ptr, m_nlabels, m_nlabels)
		),
		DIAG_KERNEL,
		NORMALIZE_SYMMETRIC
	);
}

void DenseCRFWrapper::set_unary_energy(float* unary_costs_ptr) {
	m_crf->setUnaryEnergy(
		Eigen::Map<const Eigen::MatrixXf>(
			unary_costs_ptr, m_nlabels, m_npixels)
	);
}

void DenseCRFWrapper::map(int n_iters, int* labels) {
	VectorXs labels_vec = m_crf->map(n_iters);
	for (int i = 0; i < m_npixels; i ++)
		labels[i] = labels_vec(i);
}
