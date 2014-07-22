
#include "CUSPEngine.h"
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/memory.h>
#include <cusp/copy.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <cusp/detail/device/spmv/csr_vector.h>
#include <cstdlib>
#include <iostream>
#include <cstring>


//class has on device
// CSR matrix
// dBeta vector
// dXBeta vector




void CUSPEngine::initialize() {
	
}

void CUSPEngine::loadXMatrix(const std::vector<int>& offsets, 
		const std::vector<int>& column_indices, 
		const std::vector<float>& values, int nCols) {
	
	int nOffsets = offsets.size();
	
    // Copy X to device
	
	dRow_offsets = offsets;
	dColumn_indices = column_indices;
	dValues = values;
	
	
	// Resize dBeta and dXBeta
	dBeta.resize(nCols);
	dXBeta.resize(nOffsets);

	    
}


void CUSPEngine::computeMultiplyBeta(const float* beta, int BetaLength, float* xbeta, int xBetaLength) {
    //std::cout << "Call to computeMultiply" << std::endl;
       
    thrust::copy(beta, beta + BetaLength, dBeta.begin());
 
    // y = A x
    typedef	cusp::device_memory											  MemorySpace;
    typedef int                                                           IndexType;
    typedef float                                                         ValueType;
    typedef typename cusp::csr_matrix<IndexType,ValueType,MemorySpace>    Matrix;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
    typedef typename cusp::array1d<ValueType,MemorySpace>::iterator       ValueIterator;
    typedef typename cusp::array1d_view<IndexIterator>                    IndexView;
    typedef typename cusp::array1d_view<ValueIterator>                    ValueView;
    typedef typename cusp::csr_matrix_view<IndexView,IndexView,ValueView> View;
    
    View A(xBetaLength, BetaLength, dRow_offsets[xBetaLength],
    	      cusp::make_array1d_view(dRow_offsets.begin(),    dRow_offsets.end()),
    	      cusp::make_array1d_view(dColumn_indices.begin(), dColumn_indices.end()),
    	      cusp::make_array1d_view(dValues.begin(),         dValues.end()));

    cusp::detail::device::spmv_csr_vector(A, thrust::raw_pointer_cast(&dBeta[0]), thrust::raw_pointer_cast(&dXBeta[0]));
    
    cudaMemcpy(xbeta, thrust::raw_pointer_cast(&dXBeta[0]), sizeof(float) * xBetaLength, cudaMemcpyDeviceToHost);    
        
    //thrust::copy(thrust::raw_pointer_cast(&dXBeta[0]), thrust::raw_pointer_cast(&dXBeta[0]) + xBetaLength, xbeta);
}

void CUSPEngine::solveCholesky(float* solveVector, float* solveMatrix, int n) {
    std::cout << "Call to solveCholesky" << std::endl;
	

	cudaError_t cudaStat; 
	cublasStatus_t stat; 
	cublasHandle_t handle; 
	
	float* d_a; 
	float* d_x;
	cudaStat=cudaMalloc((void**)&d_a,n*(n+1)/2*sizeof(*solveMatrix));
	cudaStat=cudaMalloc((void**)&d_x,n*sizeof(*solveVector));
	
	stat = cublasCreate(&handle);
	stat = cublasSetVector(n*(n+1)/2,sizeof(*solveMatrix),solveMatrix,1,d_a,1);
	stat = cublasSetVector(n,sizeof(*solveVector),solveVector,1,d_x,1);

	stat = cublasStpsv(handle,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,n,d_a,d_x,1);
	stat=cublasGetVector(n,sizeof(*solveVector),d_x,1,solveVector,1);
	
	for(int j=0;j<n;j++){
		printf("%9.6f",solveVector[j]);
		printf("\n"); 
		}
	//cublasStpsv('L', 'n', 'n', n, solveMatrix, solveVector, sizeof(bsccs::real));

}
