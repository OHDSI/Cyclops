/*
 * CUSPEngine.h
 *
 *  Created on: Oct 31, 2012
 *      Author: trevorshaddox
 */


#include <thrust/device_vector.h>
#include <cusp/csr_matrix.h>




class CUSPEngine {

public:

	CUSPEngine(){}

	virtual ~CUSPEngine() {}

	void initialize();

	void loadXMatrix(const std::vector<int>& offsets, const std::vector<int>& column_indices, const std::vector<float>& values, int nCols);

	void computeMultiplyBeta(const float* beta, int BetaLength, float* xbeta, int xBetaLength);

private:

	thrust::device_vector<int> dRow_offsets;
	thrust::device_vector<int> dColumn_indices;
	thrust::device_vector<float> dValues;

	thrust::device_vector<float> dBeta;
	thrust::device_vector<float> dXBeta;

	//cusp::csr_matrix<int,float,cusp::device_memory> dX;

};

