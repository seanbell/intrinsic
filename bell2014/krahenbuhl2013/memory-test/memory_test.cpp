#include <densecrf_map.h>

int main(const int argc, const char **argv) {
	int npixels = 1000;
	int nfeatures = 5;
	int nlabels = 20;

	float* unary_costs = new float[npixels * nlabels];
	float* pairwise_costs = new float[nlabels * nlabels];
	float* features = new float[npixels * nfeatures];
	int* result = new int[npixels];
	int n_crf_iters = 10;

	for (int i = 0; i < npixels * nlabels; i++)
		unary_costs[i] = i;
	for (int i = 0; i < nlabels * nlabels; i++)
		pairwise_costs[i] = i;
	for (int i = 0; i < nlabels * nfeatures; i++)
		features[i] = i;
	for (int i = 0; i < npixels; i++)
		result[i] = 0;

	densecrf_map_impl(
		unary_costs,
		pairwise_costs,
		features,
		nfeatures,
		npixels,
		nlabels,
		n_crf_iters,
		result
	);

	delete[] unary_costs;
	delete[] pairwise_costs;
	delete[] features;
	delete[] result;

	return 0;
}
