/*
 *
 * @author Marc Suchard
 */

#ifndef __GPUImplDefs__
#define __GPUImplDefs__

#ifdef HAVE_CONFIG_H
	#include "config.h"
#endif

//#define GPU_DEBUG_FLOW
//#define GPU_DEBUG_VALUES
//#define DP_DEBUG

//#define MIN_GPU
//#define REPLICATE_ON_CPU
#define PROFILE_GPU

#define REDUCE_ROW_GPU

#define GRADIENT_HESSIAN_GPU
#define GH_REDUCTION_GPU

//#define COKI_REDUCTION

/* Definition of REAL can be switched between 'double' and 'float' */
#ifdef DOUBLE_PRECISION
    #define REAL    double
	typedef double 	real;
#else
    #define REAL    float
	typedef float	real;
#endif

#define SIZE_REAL   sizeof(REAL)
#define INT         int
#define SIZE_INT    sizeof(INT)

#define IS_INDICATOR_MATRIX
#define COO_BLOCK_SIZE		256
#define MAX_THREADS 		(30 * 1024)
#define WARP_SIZE 			32
#define COO_MAX_BLOCKS      MAX_THREADS / (2 * COO_BLOCK_SIZE)
#define COO_WARPS_PER_BLOCK (COO_BLOCK_SIZE / WARP_SIZE)

#define NO_COLUMNS
#define CSR_BLOCK_SIZE		128
#define CSR_MAX_BLOCKS		MAX_THREADS / CSR_BLOCK_SIZE

#define UPDATE_XBETA_BLOCK_SIZE				16
#define COMPUTE_INTERMEDIATES_BLOCK_SIZE	128

#define CLEAR_MEMORY_BLOCK_SIZE				128

#define REDUCE_ROW_BLOCK_SIZE	64
#define MAKE_RATIO_BLOCK_SIZE	64

#define READ_ONCE

#define PAD					1		// Removes some bank conflicts (?)
#define BLOCK_SIZE_COL		16		// # of data columns to process per block
#define BLOCK_SIZE_ROW 		128		// BLOCK_SIZE_ROW / HALFWARP = # of rows (components) to process per block
#define HALFWARP_LOG2		4
#define HALFWARP 			(1<<HALFWARP_LOG2)
#define GROW_INDICES		16
#define COMPACT_BLOCK		256

#define BLOCK_SIZE_REDUCE_ALL 128


#define MEMCNV(to, from, length, toType)    { \
                                                int m; \
                                                for(m = 0; m < length; m++) { \
                                                    to[m] = (toType) from[m]; \
                                                } \
                                            }

typedef struct Dim3Int Dim3Int;

struct Dim3Int
{
    unsigned int x, y, z;
#if defined(__cplusplus)
    Dim3Int(unsigned int x = 1,
            unsigned int y = 1,
            unsigned int z = 1) : x(x), y(y), z(z) {}
#endif /* __cplusplus */
};

#endif // __GPUImplDefs__
