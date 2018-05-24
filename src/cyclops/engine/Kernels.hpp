/*
 * Kernels.hpp
 *
 *  Created on: Apr 4, 2016
 *      Author: msuchard
 */

#ifndef KERNELS_HPP_
#define KERNELS_HPP_

#include <boost/compute/type_traits/type_name.hpp>
#include "ModelSpecifics.h"

namespace bsccs {

namespace {

template<typename T, bool isNvidiaDevice>
struct ReduceBody1 {
    static std::string body() {
        std::stringstream k;
        // local reduction for non-NVIDIA device
        k <<
            "   for(int j = 1; j < TPB; j <<= 1) {        \n" <<
            "       barrier(CLK_LOCAL_MEM_FENCE);         \n" <<
            "       uint mask = (j << 1) - 1;             \n" <<
            "       if ((lid & mask) == 0) {              \n" <<
            "           scratch[lid] += scratch[lid + j]; \n" <<
            "       }                                     \n" <<
            "   }                                         \n";
        return k.str();
    }
};

template<typename T, bool isNvidiaDevice>
struct ReduceBody2 {
    static std::string body() {
        std::stringstream k;
        // local reduction for non-NVIDIA device
        k <<
            "   for(int j = 1; j < TPB; j <<= 1) {          \n" <<
            "       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
            "       uint mask = (j << 1) - 1;               \n" <<
            "       if ((lid & mask) == 0) {                \n" <<
            "           scratch[0][lid] += scratch[0][lid + j]; \n" <<
            "           scratch[1][lid] += scratch[1][lid + j]; \n" <<
            "       }                                       \n" <<
            "   }                                           \n";
        return k.str();
    }
};

template<typename T>
struct ReduceBody1<T,true>
{
    static std::string body() {
        std::stringstream k;
        k <<
            "   barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "   if (TPB >= 1024) { if (lid < 512) { sum += scratch[lid + 512]; scratch[lid] = sum; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
            "   if (TPB >=  512) { if (lid < 256) { sum += scratch[lid + 256]; scratch[lid] = sum; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
            "   if (TPB >=  256) { if (lid < 128) { sum += scratch[lid + 128]; scratch[lid] = sum; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
            "   if (TPB >=  128) { if (lid <  64) { sum += scratch[lid +  64]; scratch[lid] = sum; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
        // warp reduction
            "   if (lid < 32) { \n" <<
        // volatile this way we don't need any barrier
            "       volatile __local TMP_REAL *lmem = scratch;                 \n" <<
            "       if (TPB >= 64) { lmem[lid] = sum = sum + lmem[lid + 32]; } \n" <<
            "       if (TPB >= 32) { lmem[lid] = sum = sum + lmem[lid + 16]; } \n" <<
            "       if (TPB >= 16) { lmem[lid] = sum = sum + lmem[lid +  8]; } \n" <<
            "       if (TPB >=  8) { lmem[lid] = sum = sum + lmem[lid +  4]; } \n" <<
            "       if (TPB >=  4) { lmem[lid] = sum = sum + lmem[lid +  2]; } \n" <<
            "       if (TPB >=  2) { lmem[lid] = sum = sum + lmem[lid +  1]; } \n" <<
            "   }                                                            \n";
        return k.str();
    }
};

template<typename T>
struct ReduceBody2<T,true>
{
    static std::string body() {
        std::stringstream k;
        k <<
            "   barrier(CLK_LOCAL_MEM_FENCE); \n" <<
            "   if (TPB >= 1024) { if (lid < 512) { sum0 += scratch[0][lid + 512]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 512]; scratch[1][lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
            "   if (TPB >=  512) { if (lid < 256) { sum0 += scratch[0][lid + 256]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 256]; scratch[1][lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
            "   if (TPB >=  256) { if (lid < 128) { sum0 += scratch[0][lid + 128]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 128]; scratch[1][lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
            "   if (TPB >=  128) { if (lid <  64) { sum0 += scratch[0][lid +  64]; scratch[0][lid] = sum0; sum1 += scratch[1][lid +  64]; scratch[1][lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
        // warp reduction
            "   if (lid < 32) { \n" <<
        // volatile this way we don't need any barrier
            "       volatile __local TMP_REAL **lmem = scratch; \n" <<
            "       if (TPB >= 64) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+32]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+32]; } \n" <<
            "       if (TPB >= 32) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+16]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+16]; } \n" <<
            "       if (TPB >= 16) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 8]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 8]; } \n" <<
            "       if (TPB >=  8) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 4]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 4]; } \n" <<
            "       if (TPB >=  4) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 2]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 2]; } \n" <<
            "       if (TPB >=  2) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 1]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 1]; } \n" <<
            "   }                                            \n";
        return k.str();
    }
};

template <class BaseModel> // if BaseModel derives from IndependentData
typename std::enable_if<std::is_base_of<IndependentData,BaseModel>::value, std::string>::type
static group(const std::string& id, const std::string& k) {
    return k;
};

template <class BaseModel> // if BaseModel does not derive from IndependentData
typename std::enable_if<!std::is_base_of<IndependentData,BaseModel>::value, std::string>::type
static group(const std::string& id, const std::string& k) {
    return id + "[" + k + "]";
};

/*
static std::string timesX(const std::string& arg, const FormatType formatType) {
    return (formatType == INDICATOR || formatType == INTERCEPT) ?
        arg : arg + " * x";
}

static std::string weight(const std::string& arg, bool useWeights) {
    return useWeights ? "w * " + arg : arg;
}
*/


}; // anonymous namespace

	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForGradientHessianKernel(FormatType formatType, bool useWeights, bool isNvidia) {

        std::string name = "computeGradHess" + getFormatTypeExtension(formatType) + (useWeights ? "W" : "N");

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(            \n" <<
                "       const uint offX,                  \n" <<
                "       const uint offK,                  \n" <<
                "       const uint N,                     \n" <<
                "       __global const REAL* X,           \n" <<
                "       __global const int* K,            \n" <<
                "       __global const REAL* Y,           \n" <<
                "       __global const REAL* xBeta,       \n" <<
                "       __global const REAL* expXBeta,    \n" <<
                "       __global const REAL* denominator, \n" <<
#ifdef USE_VECTOR
                "       __global TMP_REAL* buffer,     \n" <<
#else
                "       __global REAL* buffer,            \n" <<
#endif // USE_VECTOR
                "       __global const int* id,           \n" <<  // TODO Make id optional
                "       __global const REAL* weight) {    \n";    // TODO Make weight optional

        // Initialization
        code << "   const uint lid = get_local_id(0); \n" <<
                "   const uint loopSize = get_global_size(0); \n" <<
                "   uint task = get_global_id(0);  \n" <<
                    // Local and thread storage
#ifdef USE_VECTOR
                "   __local TMP_REAL scratch[TPB]; \n" <<
                "   TMP_REAL sum = 0.0;            \n" <<
#else
                "   __local REAL scratch[2][TPB];  \n" <<
               // "   __local REAL scratch1[TPB];  \n" <<
                "   REAL sum0 = 0.0; \n" <<
                "   REAL sum1 = 0.0; \n" <<
#endif // USE_VECTOR
                    //
                "   while (task < N) { \n";

        // Fused transformation-reduction

        if (formatType == INDICATOR || formatType == SPARSE) {
            code << "       const uint k = K[offK + task];         \n";
        } else { // DENSE, INTERCEPT
            code << "       const uint k = task;            \n";
        }

        if (formatType == SPARSE || formatType == DENSE) {
            code << "       const REAL x = X[offX + task]; \n";
        } else { // INDICATOR, INTERCEPT
            // Do nothing
        }

        code << "       const REAL exb = expXBeta[k];     \n" <<
                "       const REAL numer = " << timesX("exb", formatType) << ";\n" <<
                //"       const REAL denom = 1.0 + exb;			\n";
        		"		const REAL denom = denominator[k];		\n";
        if (useWeights) {
            code << "       const REAL w = weight[k];\n";
        }

/*
        code << "       const REAL g = numer / denom;      \n";
        code << "       const REAL gradient = " << weight("g", useWeights) << ";\n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       const REAL hessian  = " << weight("g * ((REAL)1.0 - g)", useWeights) << ";\n";
        } else {
            code << "       const REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
                    "       const REAL hessian  = " << weight("(nume2 / denom - g * g)", useWeights) << ";\n";
        }
        */

        code << BaseModelG::incrementGradientAndHessianG(formatType, useWeights);

#ifdef USE_VECTOR
        code << "       sum += (TMP_REAL)(gradient, hessian); \n";
#else
        code << "       sum0 += gradient; \n" <<
                "       sum1 += hessian;  \n";
#endif // USE_VECTOR

        // Bookkeeping
        code << "       task += loopSize; \n" <<
                "   } \n" <<
                    // Thread -> local
#ifdef USE_VECTOR
                "   scratch[lid] = sum; \n";
#else
                "   scratch[0][lid] = sum0; \n" <<
                "   scratch[1][lid] = sum1; \n";
#endif // USE_VECTOR

#ifdef USE_VECTOR
        // code << (isNvidia ? ReduceBody1<real,true>::body() : ReduceBody1<real,false>::body());
        code << ReduceBody1<real,false>::body();
#else
        code << (isNvidia ? ReduceBody2<real,true>::body() : ReduceBody2<real,false>::body());
#endif

        code << "   if (lid == 0) { \n" <<
#ifdef USE_VECTOR
                "       buffer[get_group_id(0)] = scratch[0]; \n" <<
#else
                "       buffer[get_group_id(0)] = scratch[0][0]; \n" <<
                "       buffer[get_group_id(0) + get_num_groups(0)] = scratch[1][0]; \n" <<
#endif // USE_VECTOR
                "   } \n";

        code << "}  \n"; // End of kernel

        return SourceCode(code.str(), name);
    }

	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncCVGradientHessianKernel(FormatType formatType, bool isNvidia) {

		std::string name = "computeSyncCVGradHess" + getFormatTypeExtension(formatType);

		std::stringstream code;
		code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

		code << "__kernel void " << name << "(            \n" <<
				"       const uint offX,                  \n" <<
				"       const uint offK,                  \n" <<
				"       const uint N,                     \n" <<
				"       __global const REAL* X,           \n" <<
				"       __global const int* K,            \n" <<
				"       __global const REAL* Y,           \n" <<
				"       __global const REAL* xBetaVector,       \n" <<
				"       __global const REAL* expXBetaVector,    \n" <<
				"       __global const REAL* denomPidVector, \n" <<
#ifdef USE_VECTOR
				"       __global TMP_REAL* buffer,     \n" <<
#else
				"       __global REAL* buffer,            \n" <<
#endif // USE_VECTOR
				"       __global const int* pIdVector,           \n" <<  // TODO Make id optional
				"       __global const REAL* weightVector,	\n" <<
				"		const uint cvIndexStride,		\n" <<
				"		const uint blockSize,			\n" <<
				"		__global int* allZero) {   		 	\n";    // TODO Make weight optional
		// Initialization
		code << "	if (get_global_id(0) == 0) allZero[0] = 1;	\n" <<
				"	uint lid0 = get_local_id(0);		\n" <<
		        "   uint gid1, loops, remainder; 			\n" <<
				"   gid1 = get_group_id(1);             \n" <<
				"	uint loopSize = get_num_groups(1);	\n" <<
		        //"   loopSize = get_num_groups(1);       \n" <<
				"	loops = N / loopSize;					\n" <<
				"	remainder = N % loopSize;			\n" <<
				"	uint task1;							\n" <<
				"	if (gid1 < remainder) {				\n" <<
				"		task1 = gid1*(loops+1);			\n" <<
				"	} else {							\n" <<
				"		task1 = remainder*(loops+1) + (gid1-remainder)*loops;	\n" <<
				"	}									\n" <<
				//"	uint task1 = gid1;					\n" <<
				//"	__local REAL scratch[2][TPB];		\n" <<
		        //"   __local REAL sum0[TPB];             \n" <<
		        //"   __local REAL sum1[TPB];             \n" <<
				//"	sum0[lid0] = 0.0;					\n" <<
				//"	sum1[lid0] = 0.0;					\n" <<
				"	uint cvIndex = get_group_id(0)*blockSize+lid0;	\n" <<
				"	REAL sum0 = 0.0;					\n" <<
				"	REAL sum1 = 0.0;					\n";

		code <<	"	for (int i=0; i<loops; i++) {		\n";
		if (formatType == INDICATOR || formatType == SPARSE) {
			code << "  	uint k = K[offK + task1];      	\n";
		} else { // DENSE, INTERCEPT
			code << "   uint k = task1;           		\n";
		}
		if (formatType == SPARSE || formatType == DENSE) {
			code << "  	REAL x = X[offX + task1]; \n";
		} else { // INDICATOR, INTERCEPT
			// Do nothing
		}
		code << "		uint vecOffset = k*cvIndexStride + cvIndex;	\n" <<
				"		REAL xb = xBetaVector[vecOffset];			\n" <<
				"		REAL exb = exp(xb);							\n" <<
				//"		REAL exb = expXBetaVector[vecOffset];	\n" <<
				"		REAL numer = " << timesX("exb", formatType) << ";\n" <<
				"		REAL denom = (REAL)1.0 + exb;				\n" <<
				//"		REAL denom = denomPidVector[vecOffset];		\n" <<
				"		REAL w = weightVector[vecOffset];\n";
		code << BaseModelG::incrementGradientAndHessianG(formatType, true);
		code << "       sum0 += gradient; \n" <<
				"       sum1 += hessian;  \n";
		code << "       task1 += 1; \n" <<
				"   } \n";

		code << "	if (gid1 < remainder)	{				\n";
		if (formatType == INDICATOR || formatType == SPARSE) {
			code << "  	uint k = K[offK + task1];      	\n";
		} else { // DENSE, INTERCEPT
			code << "   uint k = task1;           		\n";
		}
		if (formatType == SPARSE || formatType == DENSE) {
			code << "  	REAL x = X[offX + task1]; \n";
		} else { // INDICATOR, INTERCEPT
			// Do nothing
		}
		code << "		uint vecOffset = k*cvIndexStride + cvIndex;	\n" <<
				"		REAL xb = xBetaVector[vecOffset];			\n" <<
				"		REAL exb = exp(xb);							\n" <<
				//"		REAL exb = expXBetaVector[vecOffset];	\n" <<
				"		REAL numer = " << timesX("exb", formatType) << ";\n" <<
				"		REAL denom = (REAL)1.0 + exb;				\n" <<
				//"		REAL denom = denomPidVector[vecOffset];		\n" <<
				"		REAL w = weightVector[vecOffset];\n";
		code << BaseModelG::incrementGradientAndHessianG(formatType, true);
		code << "       sum0 += gradient; \n" <<
				"       sum1 += hessian;  \n" <<
				"	}						\n";

		code << "	buffer[cvIndexStride*gid1 + cvIndex] = sum0;	\n" <<
				"	buffer[cvIndexStride*(gid1+loopSize) + cvIndex] = sum1;	\n" <<
				"	}									\n";
		//code << "}  \n"; // End of kernel
		return SourceCode(code.str(), name);
	}

	/*
		template <class BaseModel, typename WeightType, class BaseModelG>
	    SourceCode
	    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncCVGradientHessianKernel(FormatType formatType, bool isNvidia) {

			std::string name = "computeSyncCVGradHess" + getFormatTypeExtension(formatType);

			std::stringstream code;
			code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

			code << "__kernel void " << name << "(            \n" <<
					"       const uint offX,                  \n" <<
					"       const uint offK,                  \n" <<
					"       const uint N,                     \n" <<
					"       __global const REAL* X,           \n" <<
					"       __global const int* K,            \n" <<
					"       __global const REAL* Y,           \n" <<
					"       __global const REAL* xBetaVector,       \n" <<
					"       __global const REAL* expXBetaVector,    \n" <<
					"       __global const REAL* denomPidVector, \n" <<
	#ifdef USE_VECTOR
					"       __global TMP_REAL* buffer,     \n" <<
	#else
					"       __global REAL* buffer,            \n" <<
	#endif // USE_VECTOR
					"       __global const int* pIdVector,           \n" <<  // TODO Make id optional
					"       __global const REAL* weightVector,	\n" <<
					"		const uint stride,				\n" <<
					"		const uint indexWorkSize,		\n" <<
					"		__global const int* cvIndices) {    \n";    // TODO Make weight optional
			// Initialization
			code << "   uint lid = get_local_id(0); \n" <<
					"   uint task = get_global_id(0)%indexWorkSize;  \n" <<
					"	uint bufferIndex = get_global_id(0)/indexWorkSize;	\n" <<
					"	uint vecOffset = cvIndices[bufferIndex] * stride;	\n" <<
					//"	__local uint bufferIndex, cvIndex, vecOffset;	\n" <<
					//"	bufferIndex = get_group_id(0)/wgs;	\n" <<
					//"	cvIndex = cvIndices[bufferIndex]; \n" <<
					//"	vecOffset = stride*cvIndex;	\n" <<
					// Local and thread storage
	#ifdef USE_VECTOR
					"   __local TMP_REAL scratch[TPB]; \n" <<
					"   TMP_REAL sum = 0.0;            \n" <<
	#else
					"   __local REAL scratch[2][TPB];  \n" <<
					"	scratch[0][lid] = 0;			\n" <<
					"	scratch[1][lid] = 0;			\n" <<
					// "   __local REAL scratch1[TPB];  \n" <<
					"   REAL sum0 = 0.0; \n" <<
					"   REAL sum1 = 0.0; \n" <<
	#endif // USE_VECTOR
					"   if (task < N) { \n";
			// Fused transformation-reduction
			if (formatType == INDICATOR || formatType == SPARSE) {
				code << "       uint k = K[offK + task];         \n";
			} else { // DENSE, INTERCEPT
				code << "       uint k = task;            \n";
			}
			if (formatType == SPARSE || formatType == DENSE) {
				code << "       REAL x = X[offX + task]; \n";
			} else { // INDICATOR, INTERCEPT
				// Do nothing
			}
			code << "       REAL exb = expXBetaVector[vecOffset+k];     \n" <<
					"       REAL numer = " << timesX("exb", formatType) << ";\n" <<
					//"       const REAL denom = 1.0 + exb;			\n";
					"		REAL denom = denomPidVector[vecOffset+k];		\n" <<
					"       REAL w = weightVector[vecOffset+k];\n";
			code << BaseModelG::incrementGradientAndHessianG(formatType, true);
			code << "       sum0 += gradient; \n" <<
					"       sum1 += hessian;  \n";
			// Bookkeeping
			code << "       task += indexWorkSize; \n" <<
					"   } \n" <<
					// Thread -> local
					"   scratch[0][lid] = sum0; \n" <<
					"   scratch[1][lid] = sum1; \n";
			code << (isNvidia ? ReduceBody2<real,true>::body() : ReduceBody2<real,false>::body());
			code << "   if (lid == 0) { \n" <<
					"		buffer[get_group_id(0)]	= scratch[0][0];	\n" <<
					"		buffer[get_group_id(0) + get_num_groups(0)] = scratch[1][0];	\n" <<
					//"       buffer[get_group_id(0)%wgs + bufferIndex*wgs*2] = scratch[0][0]; \n" <<
					//"       buffer[get_group_id(0)%wgs + bufferIndex*wgs*2+wgs] = scratch[1][0]; \n" <<
					"   } \n";
			code << "}  \n"; // End of kernel
			return SourceCode(code.str(), name);
		}
		*/

	template <class BaseModel, typename WeightType, class BaseModelG>
	SourceCode
	GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncCV1GradientHessianKernel(FormatType formatType, bool isNvidia) {

		std::string name = "computeGradHessSync1" + getFormatTypeExtension(formatType);

		std::stringstream code;
		code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

		code << "__kernel void " << name << "(            \n" <<
				"       __global const uint* offXVec,                  \n" <<
				"       __global const uint* offKVec,                  \n" <<
				"       __global const uint* NVec,                     \n" <<
				"       __global const REAL* X,           \n" <<
				"       __global const int* K,            \n" <<
				"       __global const REAL* Y,           \n" <<
				"       __global const REAL* xBetaVector,       \n" <<
				"       __global const REAL* expXBetaVector,    \n" <<
				"       __global const REAL* denomPidVector, \n" <<
#ifdef USE_VECTOR
				"       __global TMP_REAL* buffer,     \n" <<
#else
				"       __global REAL* buffer,            \n" <<
#endif // USE_VECTOR
				"       __global const int* pIdVector,           \n" <<  // TODO Make id optional
				"       __global const REAL* weightVector,	  \n" <<
				"		const uint stride,				\n" <<
				"		const uint indexWorkSize,		\n" <<
				"		const uint wgs,					\n" <<
				"		__global const int* indices,	\n" <<
				"		__global const int* cvIndices) {    \n";    // TODO Make weight optional
		// Initialization
		code << "   uint lid = get_local_id(0); \n" <<
				"   uint task = get_global_id(0)%indexWorkSize;  \n" <<
				//"	uint bufferIndex = get_group_id(0)/wgs;	\n" <<
				//"	uint vecOffset = stride * cvIndices[bufferIndex];	\n" <<
				"	__local uint bufferIndex, index, vecOffset, N, offX, offK;	\n" <<
				"	bufferIndex = get_group_id(0)/wgs;	\n" <<
				"	index = indices[bufferIndex];		\n" <<
				//"	cvIndex = cvIndices[bufferIndex]; 	\n" <<
				"	vecOffset = stride*cvIndices[bufferIndex];			\n" <<
				"	offX = offXVec[index];				\n" <<
				"	offK = offKVec[index];				\n" <<
				"	N = NVec[index];					\n" <<
				// Local and thread storage
#ifdef USE_VECTOR
				"   __local TMP_REAL scratch[TPB]; \n" <<
				"   TMP_REAL sum = 0.0;            \n" <<
#else
				"   __local REAL scratch[2][TPB];  \n" <<
				"	scratch[0][lid] = 0;			\n" <<
				"	scratch[1][lid] = 0;			\n" <<
				// "   __local REAL scratch1[TPB];  \n" <<
				"   REAL sum0 = 0.0; \n" <<
				"   REAL sum1 = 0.0; \n" <<
#endif // USE_VECTOR
				"   while (task < N) { \n";
		// Fused transformation-reduction
		if (formatType == INDICATOR || formatType == SPARSE) {
			code << "       uint k = K[offK + task];         \n";
		} else { // DENSE, INTERCEPT
			code << "       uint k = task;            \n";
		}
		if (formatType == SPARSE || formatType == DENSE) {
			code << "       REAL x = X[offX + task]; \n";
		} else { // INDICATOR, INTERCEPT
			// Do nothing
		}
		code << "       REAL exb = expXBetaVector[vecOffset+k];     \n" <<
				"       REAL numer = " << timesX("exb", formatType) << ";\n" <<
				//"       const REAL denom = 1.0 + exb;			\n";
				"		REAL denom = denomPidVector[vecOffset+k];		\n" <<
				"       REAL w = weightVector[vecOffset+k];\n";
		code << BaseModelG::incrementGradientAndHessianG(formatType, true);

		code << "       sum0 += gradient; \n" <<
				"       sum1 += hessian;  \n";
		// Bookkeeping
		code << "       task += indexWorkSize; \n" <<
				"   } \n" <<
				// Thread -> local
				"   scratch[0][lid] = sum0; \n" <<
				"   scratch[1][lid] = sum1; \n";
		code << (isNvidia ? ReduceBody2<real,true>::body() : ReduceBody2<real,false>::body());
		code << "   if (lid == 0) { \n" <<
				"       buffer[get_group_id(0)] = scratch[0][0]; \n" <<
				"       buffer[get_group_id(0)+get_num_groups(0)] = scratch[1][0]; \n" <<
				//"       buffer[get_group_id(0)%wgs + bufferIndex*wgs*2] = scratch[0][0]; \n" <<
				//"       buffer[get_group_id(0)%wgs + bufferIndex*wgs*2+wgs] = scratch[1][0]; \n" <<
				"   } \n";
		code << "}  \n"; // End of kernel
		return SourceCode(code.str(), name);
	}

	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
	GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForMMGradientHessianKernel(FormatType formatType, bool useWeights, bool isNvidia) {

        std::string name = "computeMMGradHess" + getFormatTypeExtension(formatType) + (useWeights ? "W" : "N");

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(            \n" <<
                "       __global const uint* offXVec,                  \n" <<
                "       __global const uint* offKVec,                  \n" <<
                "       __global const uint* NVec,                     \n" <<
                //"       const uint offX,                  \n" <<
                //"       const uint offK,                  \n" <<
                //"       const uint N,                     \n" <<
                "       __global const REAL* X,           \n" <<
                "       __global const int* K,            \n" <<
                "       __global const REAL* Y,           \n" <<
                "       __global const REAL* xBeta,       \n" <<
                "       __global const REAL* expXBeta,    \n" <<
                "       __global const REAL* denominator, \n" <<
#ifdef USE_VECTOR
                "       __global TMP_REAL* buffer,     \n" <<
#else
                "       __global REAL* buffer,            \n" <<
#endif // USE_VECTOR
                "       __global const int* id,           \n" <<  // TODO Make id optional
                "       __global const REAL* weight,	  \n" <<
				"		__global const REAL* norm,		  \n" <<
				//"		const uint index) {\n";
				"		const uint indexWorkSize,					\n" <<
				"		const uint wgs,					\n" <<
				"		__global const int* indices) {    \n";    // TODO Make weight optional
        // Initialization
        code << "   const uint lid = get_local_id(0); \n" <<
        		"	__local uint bufferIndex, index, offX, offK, N;	\n" <<
        		//"   const uint loopSize = get_global_size(0); \n" <<
        		//"   uint task = get_global_id(0);  \n" <<
				"	bufferIndex = get_group_id(0)/wgs;		\n" <<
				"	index = indices[bufferIndex];		\n" <<
                "   uint task = get_global_id(0)%indexWorkSize;  \n" <<
				"	offX = offXVec[index];			\n" <<
				"	offK = offKVec[index];			\n" <<
				"	N = NVec[index];				\n" <<
                    // Local and thread storage
#ifdef USE_VECTOR
                "   __local TMP_REAL scratch[TPB]; \n" <<
                "   TMP_REAL sum = 0.0;            \n" <<
#else
                "   __local REAL scratch[2][TPB];  \n" <<
				"   REAL sum0 = 0.0; \n" <<
                "   REAL sum1 = 0.0; \n" <<
				//"	if (lid == 0) printf(\"index: %d N: %d \", index, N); \n" <<
#endif // USE_VECTOR
                "   while (task < N) { \n";
        // Fused transformation-reduction
        if (formatType == INDICATOR || formatType == SPARSE) {
            code << "       const uint k = K[offK + task];         \n";
        } else { // DENSE, INTERCEPT
            code << "       const uint k = task;            \n";
        }
        if (formatType == SPARSE || formatType == DENSE) {
            code << "       const REAL x = X[offX + task]; \n";
        } else { // INDICATOR, INTERCEPT
            // Do nothing
        }
        code << "       REAL exb = expXBeta[k];     	\n" <<
        		"		REAL xb = xBeta[k];			\n" <<
				"		REAL norm0 = norm[k];				\n" <<
                "       REAL numer = " << timesX("exb", formatType) << ";\n" <<
				"		REAL denom = denominator[k];	\n";
				//"		const REAL factor = norm[k]/abs(x);				\n" <<
                //"       const REAL g = numer / denom;      \n";
        if (useWeights) {
            code << "       const REAL w = weight[k];\n";
        }
        code << "       REAL gradient = " << weight("numer / denom", useWeights) << ";\n";
        code << "		REAL hessian = 0.0;			\n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       hessian  = gradient  * norm0 / denom;				\n";
            //code << "       hessian  = " << weight("numer*norm0/denom/denom",useWeights) << ";\n";
        } else {
        	code << "if (x != 0.0) { \n" <<
        			//"		REAL nume2 = " << timesX("gradient", formatType) << "\n" <<
					//"		hessian = nume2 * norm0 / fabs(x) / denom " <<
        			"       const REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
					"       hessian  = " << weight("nume2 * norm0 / fabs(x) / denom / denom", useWeights) << ";\n" <<
            		"} \n";
        }

#ifdef USE_VECTOR
        code << "       sum += (TMP_REAL)(gradient, hessian); \n";
#else
        code << "       sum0 += gradient; \n" <<
                "       sum1 += hessian;  \n";
#endif // USE_VECTOR
        // Bookkeeping
        code << "       task += indexWorkSize; \n" <<
                "   } \n" <<
                    // Thread -> local
#ifdef USE_VECTOR
                "   scratch[lid] = sum; \n";
#else
                "   scratch[0][lid] = sum0; \n" <<
                "   scratch[1][lid] = sum1; \n";
#endif // USE_VECTOR

#ifdef USE_VECTOR
        // code << (isNvidia ? ReduceBody1<real,true>::body() : ReduceBody1<real,false>::body());
        code << ReduceBody1<real,false>::body();
#else
        code << (isNvidia ? ReduceBody2<real,true>::body() : ReduceBody2<real,false>::body());
#endif

        code << "   if (lid == 0) { \n" <<
				//"	printf(\"%f %f | \", scratch[0][0], scratch[1][0]); \n" <<
#ifdef USE_VECTOR
                "       buffer[get_group_id(0)] = scratch[0]; \n" <<
#else
                "       buffer[get_group_id(0)%wgs + bufferIndex*wgs*2] = scratch[0][0]; \n" <<
                "       buffer[get_group_id(0)%wgs + bufferIndex*wgs*2+wgs] = scratch[1][0]; \n";

#endif // USE_VECTOR
        code << "}\n";

        code << "}  \n"; // End of kernel

        return SourceCode(code.str(), name);
	}

	template <class BaseModel, typename WeightType, class BaseModelG>
	    SourceCode
		GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncCVMMGradientHessianKernel(FormatType formatType, bool isNvidia) {

	        std::string name = "computeMMGradHessSync" + getFormatTypeExtension(formatType) ;

	        std::stringstream code;
	        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	        code << "__kernel void " << name << "(            \n" <<
	                "       __global const uint* offXVec,                  \n" <<
	                "       __global const uint* offKVec,                  \n" <<
	                "       __global const uint* NVec,                     \n" <<
	                "       __global const REAL* X,           \n" <<
	                "       __global const int* K,            \n" <<
	                "       __global const REAL* Y,           \n" <<
	                "       __global const REAL* xBetaVector,       \n" <<
	                "       __global const REAL* expXBetaVector,    \n" <<
	                "       __global const REAL* denomPidVector, \n" <<
	#ifdef USE_VECTOR
	                "       __global TMP_REAL* buffer,     \n" <<
	#else
	                "       __global REAL* buffer,            \n" <<
	#endif // USE_VECTOR
	                "       __global const int* pIdVector,           \n" <<  // TODO Make id optional
	                "       __global const REAL* weightVector,	  \n" <<
					"		const uint stride,				\n" <<
					"		const uint indexWorkSize,		\n" <<
					"		const uint wgs,					\n" <<
					"		__global const int* indices,	\n" <<
					"		__global const int* cvIndices,	\n" <<
					"		__global const REAL* normVector) {    \n";    // TODO Make weight optional
	        // Initialization
	        code << "   uint lid = get_local_id(0); \n" <<
	                    "   uint task = get_global_id(0)%indexWorkSize;  \n" <<
	    				"	__local uint bufferIndex, cvIndex, index, vecOffset, offX, offK, N;	\n" <<
	    				"	bufferIndex = get_group_id(0)/wgs;	\n" <<
						"	index = indices[bufferIndex];		\n" <<
	    				"	cvIndex = cvIndices[bufferIndex]; 	\n" <<
	    				"	vecOffset = stride*cvIndex;			\n" <<
						"	offX = offXVec[index];				\n" <<
						"	offK = offKVec[index];				\n" <<
						"	N = NVec[index];					\n" <<
	                    // Local and thread storage
	#ifdef USE_VECTOR
	                "   __local TMP_REAL scratch[TPB]; \n" <<
	                "   TMP_REAL sum = 0.0;            \n" <<
	#else
	                "   __local REAL scratch[2][TPB];  \n" <<
	               // "   __local REAL scratch1[TPB];  \n" <<
	                "   REAL sum0 = 0.0; \n" <<
	                "   REAL sum1 = 0.0; \n" <<

	#endif // USE_VECTOR
	                    //
	                "   while (task < N) { \n";

	        // Fused transformation-reduction

	        if (formatType == INDICATOR || formatType == SPARSE) {
	            code << "       uint k = K[offK + task];         \n";
	        } else { // DENSE, INTERCEPT
	            code << "       uint k = task;            \n";
	        }

	        if (formatType == SPARSE || formatType == DENSE) {
	            code << "       REAL x = X[offX + task]; \n";
	        } else { // INDICATOR, INTERCEPT
	            // Do nothing
	        }

	        code << "       REAL exb = expXBetaVector[vecOffset+k];     \n" <<
	                "       REAL numer = " << timesX("exb", formatType) << ";\n" <<
	                //"       const REAL denom = 1.0 + exb;			\n";
	        		"		REAL denom = denomPidVector[vecOffset+k];		\n" <<
					"		REAL norm = normVector[vecOffset+k];		\n" <<
	            	"       REAL w = weightVector[vecOffset+k];\n";

	        code << "       const REAL gradient = " << weight("numer / denom", true) << ";\n";
	        code << "		REAL hessian = 0.0;			\n";

	        if (formatType == INDICATOR || formatType == INTERCEPT) {
	            // code << "       hessian  = gradient  * norm0 / denom;				\n";
	            code << "       hessian  = " << weight("numer*norm/denom/denom",true) << ";\n";
	        } else {
	        	code << "if (x != 0.0) { \n";
	            code << "       const REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
	            		"       hessian  = " << weight("nume2 * norm / fabs(x) / denom / denom", true) << ";\n";
	            code << "} \n";
	        }

	        code << "       sum0 += gradient; \n" <<
	                "       sum1 += hessian;  \n";

	        // Bookkeeping
	        code << "       task += indexWorkSize; \n" <<
	                "   } \n" <<
	                    // Thread -> local

	                "   scratch[0][lid] = sum0; \n" <<
	                "   scratch[1][lid] = sum1; \n";


	        code << (isNvidia ? ReduceBody2<real,true>::body() : ReduceBody2<real,false>::body());

	        code << "   if (lid == 0) { \n" <<
	                "       buffer[get_group_id(0)%wgs + bufferIndex*wgs*2] = scratch[0][0]; \n" <<
	                "       buffer[get_group_id(0)%wgs + bufferIndex*wgs*2+wgs] = scratch[1][0]; \n" <<
	                "   } \n";

	        code << "}  \n"; // End of kernel

	        return SourceCode(code.str(), name);
		}

/*
	template <class BaseModel, typename WeightType, class BaseModelG>
	    SourceCode
	    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncCVMMGradientHessianKernel(FormatType formatType, bool isNvidia) {

			std::string name = "computeMMGradHessSync" + getFormatTypeExtension(formatType);

			std::stringstream code;
			code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

			code << "__kernel void " << name << "(            \n" <<
					"       __global const uint* offXVec,                  \n" <<
					"       __global const uint* offKVec,                  \n" <<
					"       __global const uint* NVec,                     \n" <<
					"       __global const REAL* X,           \n" <<
					"       __global const int* K,            \n" <<
					"       __global const REAL* Y,           \n" <<
					"       __global const REAL* xBetaVector,       \n" <<
					"       __global const REAL* expXBetaVector,    \n" <<
					"       __global const REAL* denomPidVector, \n" <<
	#ifdef USE_VECTOR
					"       __global TMP_REAL* buffer,     \n" <<
	#else
					"       __global REAL* buffer,            \n" <<
	#endif // USE_VECTOR
					"       __global const int* pIdVector,           \n" <<  // TODO Make id optional
					"       __global const REAL* weightVector,	\n" <<
					"		const uint cvIndexStride,		\n" <<
					"		const uint size0,				\n" <<
					"		const uint syncCVFolds,			\n" <<
					"		const uint wgs,					\n" <<
					"		__global const uint* indices) {   		 	\n";    // TODO Make weight optional
			// Initialization
			code << "	uint lid0 = get_local_id(0);		\n" <<
					//"	uint task0 = get_group_id(0)*size0+lid0;	\n" <<
					"	uint lid1 = get_local_id(1);		\n" <<
					"	uint task = lid1;					\n" <<
					"	uint index = indices[get_group_id(0)/wgs];	\n" <<
					"	uint cvIndex = get_group_id(0)%wgs*size0+lid0;	\n" <<
					"	__local uint offK, offX, N, loopSize;			\n" <<
					"	loopSize = get_global_size(1);		\n" <<
					"	offK = offKVec[index];				\n" <<
					"	offX = offXVec[index];				\n" <<
					"	N = NVec[index];					\n" <<
					"	REAL sum0 = 0.0;					\n" <<
					"	REAL sum1 = 0.0;					\n" <<
					"	__local REAL grad[8][32];			\n" <<
					"	__local REAL hess[8][32];			\n" <<
					"	if (cvIndex < syncCVFolds) {		\n" <<
					"	while (task < N) {					\n";
			if (formatType == INDICATOR || formatType == SPARSE) {
				code << "  	uint k = K[offK + task];      	\n";
			} else { // DENSE, INTERCEPT
				code << "   uint k = task1;           		\n";
			}
			if (formatType == SPARSE || formatType == DENSE) {
				code << "  	REAL x = X[offX + task1]; \n";
			} else { // INDICATOR, INTERCEPT
				// Do nothing
			}
			code << "		uint vecOffset = k*cvIndexStride;	\n" <<
					"		REAL exb = expXBetaVector[vecOffset+cvIndex];	\n" <<
					"		REAL numer = " << timesX("exb", formatType) << ";\n" <<
					"		REAL denom = denomPidVector[vecOffset+cvIndex];		\n" <<
					"		REAL w = weightVector[vecOffset+cvIndex];\n";
			code << BaseModelG::incrementGradientAndHessianG(formatType, true);
			code << "       sum0 += gradient; \n" <<
					"       sum1 += hessian;  \n";
			code << "       task += loopSize; \n" <<
					"   } \n" <<
					"	buffer[index*loopSize*cvIndexStride + cvIndex+cvIndexStride*get_group_id(1)] = sum0;	\n" <<
					"	buffer[cvIndex+cvIndexStride*(get_group_id(1)+get_num_groups(1))] = sum1;	\n" <<
					"	}									\n";
			code << "}  \n"; // End of kernel
			return SourceCode(code.str(), name);
		}
		*/


	template <class BaseModel, typename WeightType, class BaseModelG>
	SourceCode
	GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForAllGradientHessianKernel(FormatType formatType, bool useWeights, bool isNvidia) {

	    std::string name = "computeAllGradHess" + getFormatTypeExtension(formatType) + (useWeights ? "W" : "N");

	    std::stringstream code;
	    code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	    code << "__kernel void " << name << "(            \n" <<
	            "       __global const uint* offXVec,                  \n" <<
	            "       __global const uint* offKVec,                  \n" <<
	            "       __global const uint* NVec,                     \n" <<
	            "       __global const REAL* X,           \n" <<
	            "       __global const int* K,            \n" <<
	            "       __global const REAL* Y,           \n" <<
	            "       __global const REAL* xBeta,       \n" <<
	            "       __global const REAL* expXBeta,    \n" <<
	            "       __global const REAL* denominator, \n" <<
	#ifdef USE_VECTOR
	            "       __global TMP_REAL* buffer,     \n" <<
	#else
	            "       __global REAL* buffer,            \n" <<
	#endif // USE_VECTOR
	            "       __global const int* id,           \n" <<  // TODO Make id optional
	            "       __global const REAL* weight,	  \n" <<
				//"		const uint index) {\n";
				"		const uint indexWorkSize,					\n" <<
				"		const uint wgs,					\n" <<
				"		__global const int* indices) {    \n";    // TODO Make weight optional
	    // Initialization
	    code << "   uint lid = get_local_id(0); \n" <<
	    		"	__local uint offX, offK, N;			\n" <<
	    		"   uint task = get_global_id(0)%indexWorkSize;  \n" <<
				"	uint index = indices[get_group_id(0)/wgs];		\n" <<
	            "   __local REAL scratch[2][TPB];  \n" <<
				"	offX = offXVec[index];			\n" <<
				"	offK = offKVec[index];			\n" <<
				"	N = NVec[index];					\n" <<
	                // Local and thread storage
	#ifdef USE_VECTOR
	            "   __local TMP_REAL scratch[TPB]; \n" <<
	            "   TMP_REAL sum = 0.0;            \n" <<
	#else
	            "   REAL sum0 = 0.0; \n" <<
	            "   REAL sum1 = 0.0; \n" <<
				//"	if (lid == 0) printf(\"index: %d N: %d \", index, N); \n" <<
	#endif // USE_VECTOR
	            "   while (task < N) { \n";
	    // Fused transformation-reduction
	    if (formatType == INDICATOR || formatType == SPARSE) {
	        code << "       const uint k = K[offK + task];         \n";
	    } else { // DENSE, INTERCEPT
	        code << "       const uint k = task;            \n";
	    }

	    if (formatType == SPARSE || formatType == DENSE) {
	        code << "       const REAL x = X[offX + task]; \n";
	    } else { // INDICATOR, INTERCEPT
	        // Do nothing
	    }

	    code << "       REAL exb = expXBeta[k];     	\n" <<
	    		"		REAL xb = xBeta[k];			\n" <<
	            "       REAL numer = " << timesX("exb", formatType) << ";\n" <<
				"		REAL denom = denominator[k];	\n";
	    if (useWeights) {
	        code << "       const REAL w = weight[k];\n";
	    }
	    code << BaseModelG::incrementGradientAndHessianG(formatType, useWeights);
	#ifdef USE_VECTOR
	    code << "       sum += (TMP_REAL)(gradient, hessian); \n";
	#else
	    code << "       sum0 += gradient; \n" <<
	            "       sum1 += hessian;  \n";
	#endif // USE_VECTOR

	    // Bookkeeping
	    code << "       task += indexWorkSize; \n" <<
	            "   } \n" <<
	                // Thread -> local
	#ifdef USE_VECTOR
	            "   scratch[lid] = sum; \n";
	#else
	            "   scratch[0][lid] = sum0; \n" <<
	            "   scratch[1][lid] = sum1; \n";
	#endif // USE_VECTOR

	#ifdef USE_VECTOR
	    // code << (isNvidia ? ReduceBody1<real,true>::body() : ReduceBody1<real,false>::body());
	    code << ReduceBody1<real,false>::body();
	#else
	    code << (isNvidia ? ReduceBody2<real,true>::body() : ReduceBody2<real,false>::body());
	#endif

	    code << "   if (lid == 0) { \n" <<
				//"	printf(\"%f %f | \", scratch[0][0], scratch[1][0]); \n" <<
	#ifdef USE_VECTOR
	            "       buffer[get_group_id(0)] = scratch[0]; \n" <<
	#else
				"       buffer[get_group_id(0)] = scratch[0][0]; \n" <<
				"       buffer[get_group_id(0)+get_num_groups(0)] = scratch[1][0]; \n" <<
	#endif // USE_VECTOR
	            "   } \n";
	    code << "}  \n"; // End of kernel

	    return SourceCode(code.str(), name);
	}

	template <class BaseModel, typename WeightType, class BaseModelG>
	SourceCode
	GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForGradientHessianKernelExactCLR(FormatType formatType, bool useWeights) {
		std::string name = "computeGradHess" + getFormatTypeExtension(formatType) + (useWeights ? "W" : "N");

		std::stringstream code;
		code << "REAL log_sum(REAL x, REAL y) {										\n" <<
				"	if (isinf(x)) return y;											\n" <<
				"	if (isinf(y)) return x;											\n" <<
				"	if (x > y) {											\n" <<
				"		return x + log(1 + exp(y-x));								\n" <<
				"	} else {														\n" <<
				"		return y + log(1 + exp(x-y));								\n" <<
				"	}																\n" <<
				//"	REAL z = max(x,y);												\n" <<
				//"	return z + log(exp(x-z) + exp(y-z));							\n" <<
				"}																	\n";

		        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

		        code << "__kernel void " << name << "(            	\n" <<
		                "   const uint offX,          	\n" <<
		                "   const uint offK,           \n" <<
		                "   const uint N,              \n" <<
		                "   __global const REAL* X,           		\n" <<
		                "   __global const int* K,            		\n" <<
						"	__global const uint* NtoK,				\n" <<
						"	__global const REAL* casesVec,			\n" <<
						"	__global const REAL* expXBeta,				\n" <<
						"	__global REAL* output,					\n" <<
						"	__global REAL* firstRow,				\n" <<
						"	const uint persons,						\n" <<
						"	const uint index,						\n" <<
						"	__global const uint* KStrata) {    					\n";
				code << "	uint lid = get_local_id(0);				\n" <<
						"	uint stratum, stratumStart, cases, total, controls;		\n" <<
						"	stratum = get_group_id(0);				\n" <<
						"	cases = casesVec[stratum];				\n" <<
						"	stratumStart = NtoK[stratum];			\n" <<
						"	total = NtoK[stratum+1] - stratumStart;	\n" <<
						"	controls = total - cases;				\n" <<
						"	uint offKStrata = index*get_num_groups(0) + stratum;	\n" <<
						"	__local REAL B0[2][TPB];					\n" <<
						"	__local REAL B1[2][TPB];				\n" <<
						"	__local REAL B2[2][TPB];				\n";
#ifdef USE_LOG_SUM
				code << "	B0[0][lid] = -INFINITY;					\n" <<
						"	B0[1][lid] = -INFINITY;					\n" <<
						"	B1[0][lid] = -INFINITY;					\n" <<
						"	B1[1][lid] = -INFINITY;					\n" <<
						"	B2[0][lid] = -INFINITY;					\n" <<
						"	B2[1][lid] = -INFINITY;					\n" <<
						"	if (lid == 0) {							\n" <<
						"		B0[0][lid] = 0;						\n" <<
						"		B0[1][lid] = 0;						\n" <<
						"	}										\n" <<
						"	const REAL logTwo = log((REAL)2.0);		\n";
#else
				code << "	B0[0][lid] = 0;							\n" <<
						"	B0[1][lid] = 0;							\n" <<
						"	B1[0][lid] = 0;							\n" <<
						"	B1[1][lid] = 0;							\n" <<
						"	B2[0][lid] = 0;							\n" <<
						"	B2[1][lid] = 0;							\n" <<
						"	if (lid == 0) {							\n" <<
						"		B0[0][lid] = 1;						\n" <<
						"		B0[1][lid] = 1;						\n" <<
						"	}										\n";
#endif
						//"	uint current = 0;						\n";


				code << "	uint loops;								\n" <<
						"	loops = cases / (TPB - 1);				\n" <<
						"	if (cases % (TPB - 1) > 0) {			\n" <<
						"		loops++;							\n" <<
						"	}										\n";

				// if loops == 1
				code << "if (loops == 1) {							\n" <<
				        "	uint current = 0;						\n";

				/*
				if (formatType == INDICATOR || formatType == SPARSE) {
				code << "	uint currentKIndex = 0;					\n" <<
						"	uint currentK = K[offK];				\n" <<
						"	while (currentK < stratumStart) {		\n" <<
						"		currentKIndex++;					\n" <<
						"		currentK = K[offK + currentKIndex];		\n" <<
						"	}										\n";
				}
				*/

				if (formatType == INDICATOR || formatType == SPARSE) {
				code << "	__local uint currentKIndex, currentK;	\n" <<
						"	if (lid == 0) {							\n" <<
						"		currentKIndex = KStrata[offKStrata];					\n" <<
						"		if (currentKIndex == -1) {			\n" <<
						"			currentK = -1;					\n" <<
						"		} else {							\n" <<
						"			currentK = K[offK+currentKIndex];	\n" <<
						"		}									\n" <<
						"	}										\n" <<
						"	barrier(CLK_LOCAL_MEM_FENCE);			\n";
				}

				if (formatType == INTERCEPT) {
					code << "REAL x;						\n";
				} else {
					code << "__local REAL x;				\n";
				}

				code << "	for (int col = 0; col < total; col++) {	\n" <<
						"		REAL U = expXBeta[stratumStart+col];	\n";

				if (formatType == DENSE) {
					code << "	if (lid == 0) {						\n" <<
							"		x = X[offX+stratumStart+col];			\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				if (formatType == INTERCEPT) {
#ifdef USE_LOG_SUM
					code << "	x = 0;					\n";
#else
					code << "	x = 1;					\n";
#endif
				}

				if (formatType == INDICATOR) {
					code << "	if (lid == 0) {						\n" <<
							"		if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
#ifdef USE_LOG_SUM
							"	x = -INFINITY;								\n" <<
#else
							"	x = 0;								\n" <<
#endif
							"		} else {	\n" <<
#ifdef USE_LOG_SUM
							"	x = 0;								\n" <<
#else
							"	x = 1;								\n" <<
#endif
							"			currentKIndex++;			\n" <<
							"			currentK = K[offK + currentKIndex];	\n" <<
							"		}								\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				if (formatType == SPARSE) {
					code << "	if (lid == 0) {						\n" <<
							"		if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
#ifdef USE_LOG_SUM
							"	x = -INFINITY;								\n" <<
#else
							"	x = 0;								\n" <<
#endif
							"		} else {						\n" <<
							"			x = X[offX+currentKIndex];	\n" <<
							"			currentKIndex++;			\n" <<
							"			currentK = K[offK + currentKIndex];	\n" <<
							"		}								\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				code << "		if (lid > 0 && lid <= cases) {						\n" <<
#ifdef USE_LOG_SUM
						//"			x = log(x);										\n" <<
						"			B0[current][lid] = log_sum(				   B0[1-current][lid], U+B0[1-current][lid-1]);	\n" <<
						"			B1[current][lid] = log_sum(log_sum(		   B1[1-current][lid], U+B1[1-current][lid-1]), x + U + B0[1-current][lid-1]);	\n" <<
						"			B2[current][lid] = log_sum(log_sum(log_sum(B2[1-current][lid], U+B2[1-current][lid-1]), x + U + B0[1-current][lid-1]), logTwo + x + U + B1[1-current][lid-1]);	\n" <<

#else
						"			B0[current][lid] = B0[1-current][lid] + U*B0[1-current][lid-1];	\n" <<
						"			B1[current][lid] = B1[1-current][lid] + U*B1[1-current][lid-1] + x*U*B0[1-current][lid-1];	\n" <<
						"			B2[current][lid] = B2[1-current][lid] + U*B2[1-current][lid-1] + x*U*B0[1-current][lid-1] + 2*x*U*B1[1-current][lid-1];	\n" <<
#endif
						"		}									\n" <<
						"		current = 1 - current;				\n" <<
						"		barrier(CLK_LOCAL_MEM_FENCE);		\n" <<
						"	}										\n";

				/*
				code << "	output[stratum*3*TPB + lid] = B0[1-current][lid];	\n" <<
						"	output[stratum*3*TPB + TPB + lid] = B1[1-current][lid];	\n" <<
						"	output[stratum*3*TPB + 2*TPB + lid] = B2[1-current][lid];	\n";
						*/


				code << "	if (lid == 0) {							\n" <<
						"		output[stratum*3] = B0[1-current][cases];	\n" <<
						"		output[stratum*3+1] = B1[1-current][cases];	\n" <<
						"		output[stratum*3+2] = B2[1-current][cases];	\n" <<
						"	}										\n";



				code << "} else {									\n";

				// loop 1
				code << "	int start = 0;							\n" << //loop * TPB;					\n" <<
						"	int end = start + TPB - 1 + controls;		\n";//start + TPB + controls;		\n" <<
				//code << "	if (end>total) end = total;				\n";
				if (formatType == INDICATOR || formatType == SPARSE) {
				code << "	__local uint currentKIndex, currentK;	\n" <<
						"	if (lid == 0) {							\n" <<
						"		currentKIndex = KStrata[offKStrata];	\n" <<
						"		if (currentKIndex == -1) {				\n" <<
						"			currentK = -1;						\n" <<
						"		} else {								\n" <<
						"			currentK = K[offK+currentKIndex];	\n" <<
						"		}									\n" <<
						"	}										\n" <<
						"	barrier(CLK_LOCAL_MEM_FENCE);			\n";
				}
				code << "	uint current = 0;						\n";
				if (formatType == INTERCEPT) {
					code << "REAL x;						\n";
				} else {
					code << "__local REAL x;				\n";
				}

				code << "	for (int col = start; col < end; col++) {	\n" <<
						"		REAL U = expXBeta[stratumStart+col];	\n";
				if (formatType == DENSE) {
					code << "	if (lid == 0) {						\n" <<
							"		x = X[offX+stratumStart+col];			\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				if (formatType == INTERCEPT) {
#ifdef USE_LOG_SUM
					code << "	x = 0;					\n";
#else
					code << "	x = 1;					\n";
#endif
				}

				if (formatType == INDICATOR) {
					code << "	if (lid == 0) {						\n" <<
							"		if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
#ifdef USE_LOG_SUM
							"	x = -INFINITY;								\n" <<
#else
							"	x = 0;								\n" <<
#endif
							"		} else {	\n" <<
#ifdef USE_LOG_SUM
							"	x = 0;								\n" <<
#else
							"	x = 1;								\n" <<
#endif
							"			currentKIndex++;			\n" <<
							"			currentK = K[offK + currentKIndex];	\n" <<
							"		}								\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				if (formatType == SPARSE) {
					code << "	if (lid == 0) {						\n" <<
							"		if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
#ifdef USE_LOG_SUM
							"	x = -INFINITY;								\n" <<
#else
							"	x = 0;								\n" <<
#endif
							"		} else {						\n" <<
							"			x = X[offX+currentKIndex];	\n" <<
							"			currentKIndex++;			\n" <<
							"			currentK = K[offK + currentKIndex];	\n" <<
							"		}								\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}
				code << "		if (lid > 0) {						\n" <<
#ifdef USE_LOG_SUM
						//"			x = log(x);										\n" <<
						"			B0[current][lid] = log_sum(				   B0[1-current][lid], U+B0[1-current][lid-1]);	\n" <<
						"			B1[current][lid] = log_sum(log_sum(		   B1[1-current][lid], U+B1[1-current][lid-1]), x + U + B0[1-current][lid-1]);	\n" <<
						"			B2[current][lid] = log_sum(log_sum(log_sum(B2[1-current][lid], U+B2[1-current][lid-1]), x + U + B0[1-current][lid-1]), logTwo + x + U + B1[1-current][lid-1]);	\n" <<

#else
						"			B0[current][lid] = B0[1-current][lid] + U*B0[1-current][lid-1];	\n" <<
						"			B1[current][lid] = B1[1-current][lid] + U*B1[1-current][lid-1] + x*U*B0[1-current][lid-1];	\n" <<
						"			B2[current][lid] = B2[1-current][lid] + U*B2[1-current][lid-1] + x*U*B0[1-current][lid-1] + 2*x*U*B1[1-current][lid-1];	\n" <<
#endif
						"		}									\n" <<
						"		if (lid == TPB - 1)	{					\n" <<
						"			firstRow[stratumStart + col] = B0[current][lid];	\n" <<
						"			firstRow[persons + stratumStart + col] = B1[current][lid];	\n" <<
						"			firstRow[2*persons + stratumStart + col] = B2[current][lid];	\n" <<
						"		}									\n" <<
						"		current = 1 - current;				\n" <<
						"		barrier(CLK_LOCAL_MEM_FENCE);		\n" <<
						"	}										\n";

				// middle loops
				code << "	if (loops > 2) {						\n";
				code << "	for (int loop = 1; loop < loops-1; loop++) {	\n";
				code << "	barrier(CLK_GLOBAL_MEM_FENCE);			\n";
				code << "	barrier(CLK_LOCAL_MEM_FENCE);		\n";

				code << "	start = loop * (TPB - 1);					\n" <<
						"	end = start + TPB - 1 + controls;		\n" <<
						"	current = 0;						\n";

				if (formatType == INDICATOR || formatType == SPARSE) {
				code << "	if (lid == 0) {							\n" <<
						"		currentKIndex = KStrata[offKStrata];					\n" <<
						"		if (currentKIndex == -1) {			\n" <<
						"			currentK = -1;					\n" <<
						"		} else {							\n" <<
						"			currentK = K[offK+currentKIndex];		\n" <<
						"			while (currentK < stratumStart+start && currentKIndex < N) {		\n" <<
						"				currentKIndex++;						\n" <<
						"				currentK = K[offK + currentKIndex];		\n" <<
						"			}										\n" <<
						"			if (currentK >= NtoK[stratum+1] || currentKIndex == N) {	\n" <<
						"				currentK = -1;				\n" <<
						"			}								\n" <<
						"		}									\n" <<
						"	}										\n" <<
						"	barrier(CLK_LOCAL_MEM_FENCE);			\n";
				}

#ifdef USE_LOG_SUM
				code << "	B0[0][lid] = -INFINITY;					\n" <<
						"	B0[1][lid] = -INFINITY;					\n" <<
						"	B1[0][lid] = -INFINITY;					\n" <<
						"	B1[1][lid] = -INFINITY;					\n" <<
						"	B2[0][lid] = -INFINITY;					\n" <<
						"	B2[1][lid] = -INFINITY;					\n" <<
						"	if (lid == 0) {							\n" <<
						"		B0[1][lid] = firstRow[stratumStart + start-1];			\n" <<
						"		B1[1][lid] = firstRow[persons + stratumStart + start-1];	\n" <<
						"		B2[1][lid] = firstRow[2*persons + stratumStart + start-1];	\n" <<
						"	}										\n";
#else
				code << "	B0[0][lid] = 0;							\n" <<
						"	B0[1][lid] = 0;							\n" <<
						"	B1[0][lid] = 0;							\n" <<
						"	B1[1][lid] = 0;							\n" <<
						"	B2[0][lid] = 0;							\n" <<
						"	B2[1][lid] = 0;							\n" <<
						"	if (lid == 0) {							\n" <<
						"		B0[1][lid] = firstRow[stratumStart + start-1];			\n" <<
						"		B1[1][lid] = firstRow[persons + stratumStart + start-1];	\n" <<
						"		B2[1][lid] = firstRow[2*persons + stratumStart + start-1];	\n" <<
						"	}										\n";
#endif

				code << "	barrier(CLK_GLOBAL_MEM_FENCE);			\n";
				code << "	barrier(CLK_LOCAL_MEM_FENCE);			\n";

				code << "	for (int col = start; col < end; col++) {	\n" <<
						"		REAL U = expXBeta[stratumStart+col];	\n";
				if (formatType == DENSE) {
					code << "	if (lid == 0) {						\n" <<
							"		x = X[offX+stratumStart+col];			\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				if (formatType == INTERCEPT) {
#ifdef USE_LOG_SUM
					code << "	x = 0;					\n";
#else
					code << "	x = 1;					\n";
#endif
				}

				if (formatType == INDICATOR) {
					code << "	if (lid == 0) {						\n" <<
							"		if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
#ifdef USE_LOG_SUM
							"	x = -INFINITY;								\n" <<
#else
							"	x = 0;								\n" <<
#endif
							"		} else {	\n" <<
#ifdef USE_LOG_SUM
							"	x = 0;								\n" <<
#else
							"	x = 1;								\n" <<
#endif
							"			currentKIndex++;			\n" <<
							"			currentK = K[offK + currentKIndex];	\n" <<
							"		}								\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				if (formatType == SPARSE) {
					code << "	if (lid == 0) {						\n" <<
							"		if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
#ifdef USE_LOG_SUM
							"	x = -INFINITY;								\n" <<
#else
							"	x = 0;								\n" <<
#endif
							"		} else {						\n" <<
							"			x = X[offX+currentKIndex];	\n" <<
							"			currentKIndex++;			\n" <<
							"			currentK = K[offK + currentKIndex];	\n" <<
							"		}								\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}
				code << "		if (lid == 0) {						\n" <<
						"			B0[current][lid] = firstRow[stratumStart + col];	\n" <<
						"			B1[current][lid] = firstRow[persons + stratumStart + col];	\n" <<
						"			B2[current][lid] = firstRow[2*persons + stratumStart + col];	\n" <<
						"		}									\n";
				code << "	barrier(CLK_GLOBAL_MEM_FENCE);			\n";
				code << "	barrier(CLK_LOCAL_MEM_FENCE);			\n";
				code << "		if (lid > 0) {						\n" <<
#ifdef USE_LOG_SUM
						//"			x = log(x);										\n" <<
						"			B0[current][lid] = log_sum(				   B0[1-current][lid], U+B0[1-current][lid-1]);	\n" <<
						"			B1[current][lid] = log_sum(log_sum(		   B1[1-current][lid], U+B1[1-current][lid-1]), x + U + B0[1-current][lid-1]);	\n" <<
						"			B2[current][lid] = log_sum(log_sum(log_sum(B2[1-current][lid], U+B2[1-current][lid-1]), x + U + B0[1-current][lid-1]), logTwo + x + U + B1[1-current][lid-1]);	\n" <<

#else
						"			B0[current][lid] = B0[1-current][lid] + U*B0[1-current][lid-1];	\n" <<
						"			B1[current][lid] = B1[1-current][lid] + U*B1[1-current][lid-1] + x*U*B0[1-current][lid-1];	\n" <<
						"			B2[current][lid] = B2[1-current][lid] + U*B2[1-current][lid-1] + x*U*B0[1-current][lid-1] + 2*x*U*B1[1-current][lid-1];	\n" <<
#endif
						"		}									\n" <<
						"		if (lid == TPB - 1)	{					\n" <<
						"			firstRow[stratumStart + col] = B0[current][lid];	\n" <<
						"			firstRow[persons + stratumStart + col] = B1[current][lid];	\n" <<
						"			firstRow[2*persons + stratumStart + col] = B2[current][lid];	\n" <<
						"		}									\n" <<
						"		current = 1 - current;				\n" <<
						"		barrier(CLK_LOCAL_MEM_FENCE);		\n" <<
						"	}										\n";
				code << "}											\n";
				code << "}											\n";

				// final loop
				code << "	start = (loops-1) * (TPB - 1);			\n" <<
						"	end = total;						\n" <<
						"	uint lastLid = (cases-1)%(TPB-1)+1;		\n" <<
						"	current = 0;						\n";
				code << "	barrier(CLK_GLOBAL_MEM_FENCE);			\n";

				if (formatType == INDICATOR || formatType == SPARSE) {
				code << "	if (lid == 0) {								\n" <<
						"		currentKIndex = KStrata[offKStrata];	\n" <<
						"		if (currentKIndex == -1) {				\n" <<
						"			currentK = -1;						\n" <<
						"		} else {								\n" <<
						"			currentK = K[offK+currentKIndex];	\n" <<
						"			while (currentK < stratumStart+start && currentKIndex < N) {		\n" <<
						"				currentKIndex++;						\n" <<
						"				currentK = K[offK + currentKIndex];		\n" <<
						"			}									\n" <<
						"			if (currentK >= NtoK[stratum+1] || currentKIndex == N) {	\n" <<
						"				currentK = -1;					\n" <<
						"			}									\n" <<
						"		}										\n" <<
						"	}											\n" <<
						"	barrier(CLK_LOCAL_MEM_FENCE);				\n";
				}

#ifdef USE_LOG_SUM
				code << "	B0[0][lid] = -INFINITY;					\n" <<
						"	B0[1][lid] = -INFINITY;					\n" <<
						"	B1[0][lid] = -INFINITY;					\n" <<
						"	B1[1][lid] = -INFINITY;					\n" <<
						"	B2[0][lid] = -INFINITY;					\n" <<
						"	B2[1][lid] = -INFINITY;					\n" <<
						"	if (lid == 0) {							\n" <<
						"		B0[1][lid] = firstRow[stratumStart + start-1];			\n" <<
						"		B1[1][lid] = firstRow[persons + stratumStart + start-1];	\n" <<
						"		B2[1][lid] = firstRow[2*persons + stratumStart + start-1];	\n" <<
						"	}										\n";
#else
				code << "	B0[0][lid] = 0;							\n" <<
						"	B0[1][lid] = 0;							\n" <<
						"	B1[0][lid] = 0;							\n" <<
						"	B1[1][lid] = 0;							\n" <<
						"	B2[0][lid] = 0;							\n" <<
						"	B2[1][lid] = 0;							\n" <<
						"	if (lid == 0) {							\n" <<
						"		B0[1][lid] = firstRow[stratumStart + start-1];			\n" <<
						"		B1[1][lid] = firstRow[persons + stratumStart + start-1];	\n" <<
						"		B2[1][lid] = firstRow[2*persons + stratumStart + start-1];	\n" <<
						"	}										\n";
#endif
				code << "	barrier(CLK_GLOBAL_MEM_FENCE);			\n";
				code << "	barrier(CLK_LOCAL_MEM_FENCE);		\n";

				code << "	for (int col = start; col < end; col++) {	\n" <<
						"		REAL U = expXBeta[stratumStart+col];	\n";
				if (formatType == DENSE) {
					code << "	if (lid == 0) {						\n" <<
							"		x = X[offX+stratumStart+col];			\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				if (formatType == INTERCEPT) {
#ifdef USE_LOG_SUM
					code << "	x = 0;					\n";
#else
					code << "	x = 1;					\n";
#endif
				}

				if (formatType == INDICATOR) {
					code << "	if (lid == 0) {						\n" <<
							"		if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
#ifdef USE_LOG_SUM
							"	x = -INFINITY;								\n" <<
#else
							"	x = 0;								\n" <<
#endif
							"		} else {	\n" <<
#ifdef USE_LOG_SUM
							"	x = 0;								\n" <<
#else
							"	x = 1;								\n" <<
#endif
							"			currentKIndex++;			\n" <<
							"			currentK = K[offK + currentKIndex];	\n" <<
							"		}								\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				if (formatType == SPARSE) {
					code << "	if (lid == 0) {						\n" <<
							"		if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
#ifdef USE_LOG_SUM
							"	x = -INFINITY;								\n" <<
#else
							"	x = 0;								\n" <<
#endif
							"		} else {						\n" <<
							"			x = X[offX+currentKIndex];	\n" <<
							"			currentKIndex++;			\n" <<
							"			currentK = K[offK + currentKIndex];	\n" <<
							"		}								\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				code << "		if (lid == 0) {						\n" <<
						"			B0[current][lid] = firstRow[stratumStart + col];	\n" <<
						"			B1[current][lid] = firstRow[persons + stratumStart + col];	\n" <<
						"			B2[current][lid] = firstRow[2*persons + stratumStart + col];	\n" <<
						"		}									\n";
				code << "		if (lid > 0 && lid <= lastLid) {						\n" <<
#ifdef USE_LOG_SUM
						//"			x = log(x);										\n" <<
						"			B0[current][lid] = log_sum(				   B0[1-current][lid], U+B0[1-current][lid-1]);	\n" <<
						"			B1[current][lid] = log_sum(log_sum(		   B1[1-current][lid], U+B1[1-current][lid-1]), x + U + B0[1-current][lid-1]);	\n" <<
						"			B2[current][lid] = log_sum(log_sum(log_sum(B2[1-current][lid], U+B2[1-current][lid-1]), x + U + B0[1-current][lid-1]), logTwo + x + U + B1[1-current][lid-1]);	\n" <<

#else
						"			B0[current][lid] = B0[1-current][lid] + U*B0[1-current][lid-1];	\n" <<
						"			B1[current][lid] = B1[1-current][lid] + U*B1[1-current][lid-1] + x*U*B0[1-current][lid-1];	\n" <<
						"			B2[current][lid] = B2[1-current][lid] + U*B2[1-current][lid-1] + x*U*B0[1-current][lid-1] + 2*x*U*B1[1-current][lid-1];	\n" <<
#endif
						"		}									\n" <<
						"		current = 1 - current;				\n" <<
						"		barrier(CLK_LOCAL_MEM_FENCE);		\n" <<
						"	}										\n";

				/*
				code << "	output[stratum*3*TPB + lid] = B0[1-current][lid];	\n" <<
						"	output[stratum*3*TPB + TPB + lid] = B1[1-current][lid];	\n" <<
						"	output[stratum*3*TPB + 2*TPB + lid] = B2[1-current][lid];	\n";
						*/


				code << "	if (lid == lastLid) {							\n" <<
						//"		int id = (cases - 1) % (TPB-1) + 1;			\n" <<
						//"		if (id == 0) id = TPB - 1;			\n" <<
						"		output[stratum*3] = B0[1-current][lid];	\n" <<
						"		output[stratum*3+1] = B1[1-current][lid];	\n" <<
						"		output[stratum*3+2] = B2[1-current][lid];	\n" <<
						"	}										\n";


				code << "}											\n";

		        code << "}  \n"; // End of kernel
		        return SourceCode(code.str(), name);
	}


	template <class BaseModel, typename WeightType, class BaseModelG>
		SourceCode
		GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncCVGradientHessianKernelExactCLR(FormatType formatType) {
			std::string name = "computeGradHessSyncCV";

			std::stringstream code;
			code << "REAL log_sum(REAL x, REAL y) {										\n" <<
					"	if (isinf(x)) return y;											\n" <<
					"	if (isinf(y)) return x;											\n" <<
					"	if (x > y) {											\n" <<
					"		return x + log(1 + exp(y-x));								\n" <<
					"	} else {														\n" <<
					"		return y + log(1 + exp(x-y));								\n" <<
					"	}																\n" <<
					//"	REAL z = max(x,y);												\n" <<
					//"	return z + log(exp(x-z) + exp(y-z));							\n" <<
					"}																	\n";

			code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

			code << "__kernel void " << name << "(            	\n" <<
					"   const uint offX,          	\n" <<
					"   const uint offK,           \n" <<
					"   const uint N,              \n" <<
					"   __global const REAL* X,           		\n" <<
					"   __global const int* K,            		\n" <<
					"	__global const REAL* Y,					\n" <<
					"   __global const REAL* xBetaVector,       \n" <<
					"   __global const REAL* expXBetaVector,    \n" <<
					"   __global const REAL* denomPidVector, 	\n" <<
					"	__global REAL* buffer,					\n" <<
					"	__global const int* pIdVector,			\n" <<
					"	__global const REAL* weightVector,		\n" <<
					"	const uint cvIndexStride,				\n" <<
					"	const uint blockSize,					\n" <<
					"	__global int* allZero,					\n" <<
					"	__global const uint* NtoK,				\n" <<
					"	__global const REAL* casesVec,			\n" <<
					//"	__global REAL* firstRow,				\n" <<
					"	const uint persons,						\n" <<
					"	const uint index,						\n" <<
					"	const uint totalStrata,					\n" <<
					"	__global const uint* KStrata) {    	\n";

			code << "	if (get_global_id(0) == 0) allZero[0] = 1;	\n" <<
					"	__local REAL B0[2][TPB0*TPB1];			\n" <<
					"	__local REAL B1[2][TPB0*TPB1];			\n" <<
					"	__local REAL B2[2][TPB0*TPB1];			\n" <<
					"	__local REAL localWeights[TPB0];	\n" <<
					"	uint lid0 = get_local_id(0);		\n" <<
					"	uint lid1 = get_local_id(1);		\n" <<
					"	uint mylid = lid1*TPB0+lid0;		\n" <<
					"	uint cvIndex = get_group_id(0)*blockSize+lid0;	\n" <<
					"	uint stratum = get_group_id(1);			\n" <<
					"	uint loopSize = get_num_groups(1);		\n" <<
					"	REAL grad = 0;							\n" <<
					"	REAL hess = 0;							\n" <<

					"	while (stratum < totalStrata) {		\n" <<
					"		int stratumStart = NtoK[stratum];	\n" <<
					"		uint vecOffset = stratumStart*cvIndexStride + cvIndex;	\n" <<
					"		int total = NtoK[stratum+1] - stratumStart;	\n" <<
					"		int cases = casesVec[stratum];		\n" <<
					"		int controls = total - cases;		\n" <<
					"		int offKStrata = index*totalStrata + stratum;	\n" <<
					"		if (lid1 == 0) {					\n" <<
					"			REAL temp = weightVector[vecOffset];	\n" <<
					"			localWeights[lid0] = temp;		\n" <<
					"		}									\n" <<
					"		barrier(CLK_LOCAL_MEM_FENCE);		\n";
#ifdef USE_LOG_SUM
			code << "		B0[0][mylid] = -INFINITY;					\n" <<
					"		B0[1][mylid] = -INFINITY;					\n" <<
					"		B1[0][mylid] = -INFINITY;					\n" <<
					"		B1[1][mylid] = -INFINITY;					\n" <<
					"		B2[0][mylid] = -INFINITY;					\n" <<
					"		B2[1][mylid] = -INFINITY;					\n" <<
					"		if (lid1 == 0) {							\n" <<
					"			B0[0][lid0] = 0;						\n" <<
					"			B0[1][lid0] = 0;						\n" <<
					"		}										\n" <<
					"		const REAL logTwo = log((REAL)2.0);		\n";
#else
			code << "		B0[0][mylid] = 0;							\n" <<
					"		B0[1][mylid] = 0;							\n" <<
					"		B1[0][mylid] = 0;							\n" <<
					"		B1[1][mylid] = 0;							\n" <<
					"		B2[0][mylid] = 0;							\n" <<
					"		B2[1][mylid] = 0;							\n" <<
					"		if (lid1 == 0) {							\n" <<
					"			B0[0][lid0] = 1;						\n" <<
					"			B0[1][lid0] = 1;						\n" <<
					"		}										\n";
#endif
			//"	uint current = 0;						\n";

			code << "		uint loops;								\n" <<
					"		loops = cases / (TPB1 - 1);				\n" <<
					"		if (cases % (TPB1 - 1) > 0) {			\n" <<
					"			loops++;							\n" <<
					"		}										\n";

			// if loops == 1
			//code << "	if (loops == 1) {							\n" <<
			code << "		uint current = 0;						\n";

			if (formatType == INDICATOR || formatType == SPARSE) {
				code << "	__local uint currentKIndex, currentK;	\n" <<
						"	if (lid0 == 0 && lid1 == 0) {							\n" <<
						"		currentKIndex = KStrata[offKStrata];					\n" <<
						"		if (currentKIndex == -1) {			\n" <<
						"			currentK = -1;					\n" <<
						"		} else {							\n" <<
						"			currentK = K[offK+currentKIndex];	\n" <<
						"		}									\n" <<
						"	}										\n" <<
						"	barrier(CLK_LOCAL_MEM_FENCE);			\n";
			}

			if (formatType == INTERCEPT) {
				code << "	REAL x;						\n";
			} else {
				code << "	__local REAL x;				\n";
			}

			code << "		for (int col = 0; col < total; col++) {	\n" <<
#ifdef USE_LOG_SUM
					"			REAL U = xBetaVector[vecOffset + col * cvIndexStride];	\n";
#else
					"			REAL U = exp(xBetaVector[vecOffset+col*cvIndexStride]);	\n";
#endif

			if (formatType == DENSE) {
				code << "		if (lid0 == 0 && lid1 == 0) {						\n" <<
						"			x = X[offX+stratumStart+col];			\n" <<
						"		}									\n" <<
						"		barrier(CLK_LOCAL_MEM_FENCE);		\n";
			}

			if (formatType == INTERCEPT) {
#ifdef USE_LOG_SUM
				code << "		x = 0;					\n";
#else
				code << "		x = 1;					\n";
#endif
			}

			if (formatType == INDICATOR) {
				code << "		if (lid0 == 0 && lid1 == 0) {						\n" <<
						"			if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
#ifdef USE_LOG_SUM
						"			x = -INFINITY;								\n" <<
#else
						"			x = 0;								\n" <<
#endif
						"			} else {	\n" <<
#ifdef USE_LOG_SUM
						"			x = 0;								\n" <<
#else
						"			x = 1;								\n" <<
#endif
						"			currentKIndex++;			\n" <<
						"			currentK = K[offK + currentKIndex];	\n" <<
						"			}								\n" <<
						"		}									\n" <<
						"		barrier(CLK_LOCAL_MEM_FENCE);		\n";
			}

			if (formatType == SPARSE) {
				code << "		if (lid0 == 0 && lid1 == 0) {						\n" <<
						"			if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
#ifdef USE_LOG_SUM
						"			x = -INFINITY;								\n" <<
#else
						"			x = 0;								\n" <<
#endif
						"			} else {						\n" <<
						"				x = X[offX+currentKIndex];	\n" <<
						"				currentKIndex++;			\n" <<
						"			currentK = K[offK + currentKIndex];	\n" <<
						"			}								\n" <<
						"		}									\n" <<
						"		barrier(CLK_LOCAL_MEM_FENCE);		\n";
			}

			code << "			if (lid1 > 0 && lid1 <= cases) {						\n" <<
#ifdef USE_LOG_SUM
					//"			x = log(x);										\n" <<
					"				B0[current][mylid] = log_sum(				 B0[1-current][mylid], U+B0[1-current][mylid-TPB0]);	\n" <<
					"				B1[current][mylid] = log_sum(log_sum(		 B1[1-current][mylid], U+B1[1-current][mylid-TPB0]), x + U + B0[1-current][mylid-TPB0]);	\n" <<
					"				B2[current][mylid] = log_sum(log_sum(log_sum(B2[1-current][mylid], U+B2[1-current][mylid-TPB0]), x + U + B0[1-current][mylid-TPB0]), logTwo + x + U + B1[1-current][mylid-TPB0]);	\n" <<

#else
					"				B0[current][mylid] = B0[1-current][mylid] + U*B0[1-current][mylid-TPB0];	\n" <<
					"				B1[current][mylid] = B1[1-current][mylid] + U*B1[1-current][mylid-TPB0] + x*U*B0[1-current][mylid-TPB0];	\n" <<
					"				B2[current][mylid] = B2[1-current][mylid] + U*B2[1-current][mylid-TPB0] + x*U*B0[1-current][mylid-TPB0] + 2*x*U*B1[1-current][mylid-TPB0];	\n" <<
#endif
					"			}									\n" <<
					"			current = 1 - current;				\n" <<
					"			barrier(CLK_LOCAL_MEM_FENCE);		\n" <<
					"		}										\n";

			code << "		if (lid1 == 0) {							\n" <<
					"			if (localWeights[lid0] != 0) {			\n" <<
					"				REAL value0 = B0[1-current][cases*TPB0+lid0]*localWeights[lid0];	\n" <<
					"				REAL value1 = B1[1-current][cases*TPB0+lid0]*localWeights[lid0];	\n" <<
					"				REAL value2 = B2[1-current][cases*TPB0+lid0]*localWeights[lid0];	\n" <<
#ifdef USE_LOG_SUM
					"				grad -= -exp(value1 - value0);			\n" <<
					"				hess -= exp(2*(value1-value0)) - exp(value2 - value0);	\n" <<
#else
					"				grad -= -value1/value0;					\n" <<
					"				hess -= value1*value1/value0/value0 - value2/value0;		\n" <<
#endif
					"			}									\n" <<
					"		}										\n";
			code << "		stratum += loopSize;					\n";
			code << "		barrier(CLK_LOCAL_MEM_FENCE);			\n";
			code << "		barrier(CLK_GLOBAL_MEM_FENCE);			\n";
			code << "	}											\n";
			code << "	if (lid1 == 0) {							\n" <<
					"			buffer[cvIndexStride*get_group_id(1) + cvIndex] = grad;	\n" <<
					"			buffer[cvIndexStride*(loopSize+get_group_id(1)) + cvIndex] = hess;	\n" <<
					"	}											\n";
			code << "}  \n"; // End of kernel
			return SourceCode(code.str(), name);
	}


	/*
	template <class BaseModel, typename WeightType, class BaseModelG>
	SourceCode
	GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForGradientHessianKernelExactCLR(FormatType formatType, bool useWeights, bool isNvidia) {
		std::string name = "computeGradHess" + getFormatTypeExtension(formatType) + (useWeights ? "W" : "N");

		        std::stringstream code;
		        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

		        code << "__kernel void " << name << "(            	\n" <<
		        	    "   __global REAL* B0_in,                	\n" <<
		        	    "   __global REAL* B1_in,               	\n" <<
		        		"   __global const uint* indices_in, 		\n" <<
		        		"   __global const REAL* xMatrix_in, 		\n" <<
		        		"   __global const REAL* expXMatrix_in, 	\n" <<
		        		"   __global const uint* vector1_in, 		\n" <<
		        		"   __global const uint* vector2_in, 		\n" <<
		        		"   const uint N, 							\n" <<
		        	    "   const uint col,                 		\n" <<
		                "   __global const REAL* weight,			\n" <<
					    "	__global uint* overflow0_in,           	\n" <<
					    "	__global uint* overflow1_in,				\n" <<
						"	const uint tasks) {    					\n";
				code << "   const uint i = get_global_id(0);				\n" <<
						"	if (i < tasks) {						\n" <<
						"	int stratum = indices_in[i];			\n" <<
						"	REAL x = xMatrix_in[col*(N+1)+stratum];	\n" <<
						"	REAL t = expXMatrix_in[col*(N+1)+stratum];	\n" <<
						"	__global REAL* BOld;					\n" <<
						"	__global REAL* BNew;					\n" <<
						"	__global uint* OverflowOld;				\n" <<
						"	__global uint* OverflowNew;				\n" <<
						//"	__local REAL* BOld; __local REAL* BNew;	\n" <<
						"	if (col%2==0) {							\n" <<
						"		BOld=B0_in; 						\n" <<
						"		BNew=B1_in;							\n" <<
						"		OverflowOld=overflow0_in; 			\n" <<
						"		OverflowNew=overflow1_in; 			\n" <<
						"	} else {								\n" <<
						"		BOld=B1_in;							\n" <<
						"		BNew=B0_in;							\n" <<
						"		OverflowOld=overflow1_in; 			\n" <<
						"		OverflowNew=overflow0_in; 			\n" <<
						"	}										\n" <<
						"	if (t == -1) BNew[i] = BOld[i];			\n" <<
						"	else { 									\n" <<
						"		if (i > 2) BNew[i] = BOld[i] + t*BOld[i-3] + vector1_in[i]*x*t*BOld[i-3-i%3] + 2*vector2_in[i]*x*t*BOld[i-2-i%3]; \n" <<
						" 		if (OverflowOld[stratum] == 1) BNew[i] /= 1e25;          	\n" <<
						"		if (BNew[i] > 1e25 && OverflowNew[stratum] == 0) OverflowNew[stratum] = 1;	\n" <<
						"  	}										\n" <<
						"	if (i < (N+1) && col%2 == 0) OverflowOld[i] = 0;\n" <<
						"	};										\n";
		        code << "}  \n"; // End of kernel

		        return SourceCode(code.str(), name);
	}
	*/
/*
	template <class BaseModel, typename WeightType, class BaseModelG>
	SourceCode
	GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForGradientHessianKernelExactCLR(FormatType formatType, bool useWeights, bool isNvidia) {
		std::string name = "computeGradHess" + getFormatTypeExtension(formatType) + (useWeights ? "W" : "N");

		        std::stringstream code;
		        code << "REAL log_sum(REAL x, REAL y) {										\n" <<
		        		"	if (isinf(x)) return y;											\n" <<
						"	if (isinf(y)) return x;											\n" <<
						"	REAL z = max(x,y);												\n" <<
						"	return z + log(exp(x-z) + exp(y-z));							\n" <<
						"}																	\n";
		        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

		        code << "__kernel void " << name << "(            		\n" <<
		                "       __global const uint* offXVec,           \n" <<
		                "       __global const uint* offKVec,           \n" <<
						"		__global const uint* NVec,				\n" <<
					    " 		__global const uint* NtoK,				\n" <<
		                "       __global const uint* Kcurrent,          \n" <<
		        		"   	__global const REAL* X, 				\n" <<
		                "   	__global const int* K,            		\n" <<
		        		"   	__global const REAL* expXBeta, 			\n" <<
						"		__global const REAL* NWeight,			\n" <<
						"		__global REAL* buffer,					\n" <<
		        	    "   	__global REAL* firstRow,                \n" <<
						"		uint firstRowStride,				\n" <<
		        	    "   	uint iteration,                 	\n" <<
						//"		__global REAL* B0,						\n" <<
						//"		__global REAL* B1,						\n" <<
						"		__global REAL* overflowVec,			\n" <<
					    "		uint index) {    					\n";
				code << "   uint i = get_local_id(0);				\n" <<
						"	uint stratum = get_group_id(0);		\n" <<
						"	uint strata = get_num_groups(0);		\n" <<
						"	uint subjects = NtoK[stratum+1] - NtoK[stratum]; \n" <<
						"	uint cases = NWeight[stratum];		\n" <<
						"	uint controls = subjects - cases;		\n" <<
						"	if (iteration * 32 < cases) {				\n" <<
						"	__local REAL B0[99];						\n" <<
						"	__local REAL B1[99];						\n" <<
						//"	const uint BOffs = 112 * stratum;			\n" <<
						"	uint BOffs = 0;						\n" <<
#ifdef USE_LOG_SUM
						"	B0[BOffs+3*i+3] = B0[BOffs+3*i+4] = B0[BOffs+3*i+5] = -INFINITY;		\n" <<
						"	B1[BOffs+3*i+3] = B1[BOffs+3*i+4] = B1[BOffs+3*i+5] = -INFINITY;		\n" <<
						"	const REAL logTwo = log((REAL)2.0);		\n" <<
#else
						"	B0[BOffs+3*i+3] = B0[BOffs+3*i+4] = B0[BOffs+3*i+5] = 0;		\n" <<
						"	B1[BOffs+3*i+3] = B1[BOffs+3*i+4] = B1[BOffs+3*i+5] = 0;		\n" <<
#endif
                        "   __local REAL overflow;	                    \n" <<
                        "   __local REAL cumOverflow;		            \n" <<
						"	if (i == 0) {								\n" <<
						"		overflow = 1.0;							\n" <<
						"		cumOverflow = 1.0;						\n" <<
						"	}											\n" <<
						"	const uint offX = offXVec[index];			\n" <<
						"	const uint offK = offKVec[index];			\n" <<
						"	uint taskCounts = cases%32;					\n" <<
						"	if (taskCounts == 0 || (iteration+1) * 32 < cases) taskCounts = 32; \n" <<
					    //"	__global REAL* BOld;							\n" <<
						//"	__global REAL* BNew;							\n" <<
						"	volatile __local REAL* BOld;							\n" <<
						"	volatile __local REAL* BNew;							\n" <<
						//"   for (int j = 0; j < (controls+taskCounts)/32+1; ++j) {  \n" <<
						//"   	if (32*j + i < controls+taskCounts) {   \n" <<
						//"       	overflowVec[32*j+i] = 1.0;      \n" <<
						//"       }                                       \n" <<
						//"   }                                               \n" <<
						"	for (uint task = 0; task < controls + taskCounts; task++) { \n" <<
						"		if (task % 2 == 0) {						\n" <<
						"			BOld = B0+BOffs; BNew = B1+BOffs;					\n" <<
						"		} else {									\n" <<
						"			BOld = B1+BOffs; BNew = B0+BOffs;					\n" <<
						"		} 											\n" <<
						"		const uint k = iteration * 32 + task;		\n" <<
#ifdef USE_LOG_SUM
						"		const REAL exb = log(expXBeta[NtoK[stratum] + k]);	\n" <<
						"		const REAL x = log(X[offX + NtoK[stratum] + k]);	\n" <<
#else
						"		const REAL exb = expXBeta[NtoK[stratum] + k];	\n" <<
						"		const REAL x = X[offX + NtoK[stratum] + k];	\n" <<
#endif
						"		if (i == 0 && task < controls) {	\n" <<
						"			BOld[0] = cumOverflow*firstRow[firstRowStride*stratum + 3*task]; \n" <<
						"			BOld[1] = cumOverflow*firstRow[firstRowStride*stratum + 3*task + 1]; \n" <<
						"			BOld[2] = cumOverflow*firstRow[firstRowStride*stratum + 3*task + 2]; \n" <<
						"		} 											\n" <<
#ifdef USE_LOG_SUM
						"		BNew[3*i+3] = log_sum(BOld[3*i+3],exb+BOld[3*i]);	\n" <<
						"		BNew[3*i+4] = log_sum(log_sum(BOld[3*i+4], exb+BOld[3*i+1]),x+exb+BOld[3*i]);	\n" <<
						"		BNew[3*i+5] = log_sum(log_sum(log_sum(BOld[3*i+5], exb+BOld[3*i+2]), x+exb+BOld[3*i]),logTwo+x+exb+BOld[3*i+1]);	\n" <<
#else
						"		BNew[3*i+3] = overflow*(BOld[3*i+3] + exb*BOld[3*i]);	\n" <<
						"		BNew[3*i+4] = overflow*(BOld[3*i+4] + exb*BOld[3*i+1] + x*exb*BOld[3*i]);	\n" <<
						"		BNew[3*i+5] = overflow*(BOld[3*i+5] + exb*BOld[3*i+2] + x*exb*BOld[3*i] + 2*x*exb*BOld[3*i+1]);	\n" <<
						"       if (overflow!=1.0 && i == 0) {                          \n" <<
						"       	cumOverflow *= 1e-200;                          \n" <<
						"           overflowVec[firstRowStride*stratum + task] = 1e-200;                     \n" <<
						"       }                                                       \n" <<
						"       if (BNew[3*i+3]>1e200 || BNew[3*i+4]>1e200 || BNew[3*i+5]>1e200) {      \n" <<
						"       	overflow = 1e-200;                              \n" <<
						"       } else if (i==0) {                                      \n" <<
						"       	overflow = 1.0;                                 \n" <<
						"       }                                                       \n" <<
#endif
						"		if (i == 31 && task >= 31) {				\n" <<
						"			firstRow[firstRowStride*stratum + 3*(task - 31)] = BNew[3*i+3]; \n" <<
						"			firstRow[firstRowStride*stratum + 3*(task - 31) + 1] = BNew[3*i+4]; \n" <<
						"			firstRow[firstRowStride*stratum + 3*(task - 31) + 2] = BNew[3*i+5]; \n" <<
						"		}											\n" <<
						"	}												\n" <<
						"	if (iteration * 32 + taskCounts == cases && i == taskCounts-1) {	\n" <<
						"		buffer[3*stratum] = BNew[3*i+3];	\n" <<
						"		buffer[3*stratum+1] = BNew[3*i+4]; \n" <<
						"		buffer[3*stratum+2] = BNew[3*i+5]; \n" <<
						"	}												\n" <<
#ifdef USE_LOG_SUM
#else
						"	if (i==0) {                                             \n" <<
						"   	REAL tmp = 1.0;                                         \n" <<
						"   	for (int j = controls+taskCounts-1; j >= 0; --j) {              \n" <<
						"   		firstRow[firstRowStride*stratum + 3*(j-31)] *= tmp;     \n" <<
						"  			firstRow[firstRowStride*stratum + 3*(j-31)+1] *= tmp;   \n" <<
						"       	firstRow[firstRowStride*stratum + 3*(j-31)+2] *= tmp;   \n" <<
						"       	tmp *= overflowVec[firstRowStride*stratum+j];                  \n" <<
						"       }                                                       \n" <<
						"   }                                                               \n" <<
#endif
						"	}												\n";
		        code << "}  \n"; // End of kernel
		        return SourceCode(code.str(), name);
	}
	*/


	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForUpdateXBetaKernel(FormatType formatType) {

        std::string name = "updateXBeta" + getFormatTypeExtension(formatType);

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(     \n" <<
                "       const uint offX,           \n" <<
                "       const uint offK,           \n" <<
                "       const uint N,              \n" <<
                "       const REAL delta,          \n" <<
                "       __global const REAL* X,    \n" <<
                "       __global const int* K,     \n" <<
                "       __global const REAL* Y,    \n" <<
                "       __global REAL* xBeta,      \n" <<
                "       __global REAL* expXBeta,   \n" <<
                "       __global REAL* denominator,\n" <<
                "       __global const int* id) {   \n" <<
                "   const uint task = get_global_id(0); \n";

        if (formatType == INDICATOR || formatType == SPARSE) {
            code << "   const uint k = K[offK + task];         \n";
        } else { // DENSE, INTERCEPT
            code << "   const uint k = task;            \n";
        }

        if (formatType == SPARSE || formatType == DENSE) {
            code << "   const REAL inc = delta * X[offX + task]; \n";
        } else { // INDICATOR, INTERCEPT
            code << "   const REAL inc = delta;           \n";
        }

        code << "   if (task < N) {      				\n";
        code << "       REAL xb = xBeta[k] + inc; 		\n" <<
                "       xBeta[k] = xb;                  \n";
/*
        if (BaseModel::likelihoodHasDenominator) {
            // TODO: The following is not YET thread-safe for multi-row observations
            // code << "       const REAL oldEntry = expXBeta[k];                 \n" <<
            //         "       const REAL newEntry = expXBeta[k] = exp(xBeta[k]); \n" <<
            //         "       denominator[" << group<BaseModel>("id","k") << "] += (newEntry - oldEntry); \n";
            code << "       const REAL exb = exp(xb); \n" <<
                    "       expXBeta[k] = exb;        \n" <<
					"		denominator[k] = 1.0 + exb; \n";
        	//code << "expXBeta[k] = exp(xb); \n";
        	//code << "expXBeta[k] = exp(1); \n";

            // LOGISTIC MODEL ONLY
            //                     const real t = BaseModel::getOffsExpXBeta(offs, xBeta[k], y[k], k);
            //                     expXBeta[k] = t;
            //                     denominator[k] = static_cast<real>(1.0) + t;
            //             code << "    const REAL t = 0.0;               \n" <<
            //                     "   expXBeta[k] = exp(xBeta[k]);      \n" <<
            //                     "   denominator[k] = REAL(1.0) + tmp; \n";
        }
        */

        code << "   } \n";
        code << "}    \n";

        return SourceCode(code.str(), name);
    }

	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
	GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForUpdateAllXBetaKernel(FormatType formatType, bool isNvidia) {

        std::string name = "updateAllXBeta" + getFormatTypeExtension(formatType);

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(            \n" <<
                "       __global const uint* offXVec,                  \n" <<
                "       __global const uint* offKVec,                  \n" <<
                "       __global const uint* NVec,                     \n" <<
                "       __global const REAL* allDelta,                  \n" <<
                "       __global const REAL* X,           \n" <<
                "       __global const int* K,            \n" <<
                "       __global const REAL* Y,           \n" <<
                "       __global  REAL* xBeta,       \n" <<
                "       __global const REAL* expXBeta,    \n" <<
                "       __global const REAL* denominator, \n" <<
				"		const uint indexWorkSize,		\n" <<
				"		const uint wgs,					\n" <<
				"		__global const int* fixBeta) {    \n";    // TODO Make weight optional

        // Initialization
        code << "   const uint lid = get_local_id(0); \n" <<
                "   const uint loopSize = indexWorkSize; \n" <<
                "   uint task = get_global_id(0)%indexWorkSize;  \n" <<
                    // Local and thread storage
                "   __local REAL scratch[TPB];  \n" <<
                "   REAL sum = 0.0; \n" <<
				"	const uint n = get_group_id(0)/wgs;		\n" <<
				"	const uint offX = offXVec[n];			\n" <<
				"	const uint offK = offKVec[n];			\n" <<
				"	const uint N = NVec[n];					\n";//<<
				/*
        code << "   const uint lid = get_local_id(0); \n" <<
        		"	const uint n = get_group_id(0);		\n" <<
				//"	if (fixBeta[index] == 0) {					\n" <<
        		//"   uint task = get_global_id(0)%TPB; \n" <<
                //"   const uint loopSize = get_global_size(0)/J; \n" <<
				"	const uint offX = offXVec[n];			\n" <<
				"	const uint offK = offKVec[n];			\n" <<
				"	const uint N = NVec[n];					\n" <<
				//"	const REAL delta = allDelta[index];			\n" <<
                    // Local and thread storage
                "   REAL sum = 0.0; \n" <<
				"   __local REAL scratch[TPB];  \n" <<
				*/
        code << "   while (task < N) { 						\n";// <<
        if (formatType == INDICATOR || formatType == SPARSE) {
        	code << "   const uint k = K[offK + task];         \n";
        } else { // DENSE, INTERCEPT
        	code << "   const uint k = task;            \n";
        }
        code << "	const REAL delta = allDelta[k];		\n";
        if (formatType == SPARSE || formatType == DENSE) {
        	code << "   const REAL inc = delta * X[offX + task]; \n";
        } else { // INDICATOR, INTERCEPT
        	code << "   const REAL inc = delta;           \n";
        }
        //code << "	if (inc != 0.0) xBeta[k] += inc; \n";// <<
        code << " sum += inc;\n";
        //code << "if (inc != 0.0) atomic_xchg((volatile __global REAL *)(xBeta+k), xb + inc);\n";

        // Bookkeeping
        code << "       task += TPB; \n" <<
                "   } \n";// <<
                    // Thread -> local
        code << "   scratch[lid] = sum; \n";
        code << (isNvidia ? ReduceBody1<real,true>::body() : ReduceBody1<real,false>::body());

        code << "   if (lid == 0) { \n" <<
                "       xBeta[n] += scratch[0]; \n" <<
                "   } \n";
        code << "}  \n"; // End of kernel

        return SourceCode(code.str(), name);
	}


	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncUpdateXBetaKernel(FormatType formatType) {

        std::string name = "updateXBetaSync" + getFormatTypeExtension(formatType);

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(     \n" <<
                "       const uint offX,           \n" <<
                "       const uint offK,           \n" <<
                "       __global const REAL* deltaVector,          \n" <<
                "       __global const REAL* X,    \n" <<
                "       __global const int* K,     \n" <<
                "       __global const REAL* Y,    \n" <<
                "       __global REAL* xBetaVector,      \n" <<
                "       __global REAL* expXBetaVector,   \n" <<
                "       __global REAL* denomPidVector,\n" <<
                "       __global const int* id,		\n" <<
				"		const uint cvIndexStride,			\n" <<
				"		__global const REAL* Offs,	\n" <<
				"		const uint size0,			\n" <<
				"		const uint syncCVFolds,		\n" <<
				"		const uint index,			\n" <<
				"		__global const int* allZero) {   \n";

        code << "	uint lid0 = get_local_id(0);		\n" <<
        		"	if (allZero[0] == 0) {				\n" <<
        		//"	uint task0 = get_group_id(0)*size0+lid0;	\n" <<
				"	__local uint task1; 				\n" <<
        		"	task1 = get_group_id(1);			\n" <<
				"	uint cvIndex = get_group_id(0)*size0+lid0;	\n" <<
				//"	__local y, offs;					\n";// <<
				//"	if (cvIndex < syncCVFolds) {		\n" <<
				//	"		REAL delta = deltaVector[cvIndex];	\n";
				"		REAL delta = deltaVector[index*cvIndexStride+cvIndex];	\n";// <<
				//"	if (delta != 0) {					\n";
        if (formatType == INDICATOR || formatType == SPARSE) {
        	code << "  	uint k = K[offK + task1];      	\n";
        } else { // DENSE, INTERCEPT
        	code << "   uint k = task1;           		\n";
        }
        if (formatType == SPARSE || formatType == DENSE) {
            code << "   REAL inc = delta * X[offX + task1]; \n";
        } else { // INDICATOR, INTERCEPT
            code << "   REAL inc = delta;           \n";
        }
        code << "		uint vecOffset = k*cvIndexStride + cvIndex;	\n" <<
        		"		REAL xb = xBetaVector[vecOffset] + inc;	\n" <<
				"		xBetaVector[vecOffset] = xb;	\n";
        if (BaseModel::likelihoodHasDenominator) {
        	//code << "	REAL y = Y[k];\n" <<
        	//		"	REAL offs = Offs[k];\n";
        	//code << "	REAL exb = " << BaseModelG::getOffsExpXBetaG() << ";\n";
        	//code << "	expXBetaVector[vecOffset] = exb;\n";
        	//code << "	denomPidVector[vecOffset] =" << BaseModelG::getDenomNullValueG() << "+ exb;\n";
        }
        //code << "   } \n";
        //code << "}	\n";
        code << "}    \n";
        code << "}		\n";

        return SourceCode(code.str(), name);
    }


	/*
	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncUpdateXBetaKernel(FormatType formatType) {

        std::string name = "updateXBetaSync" + getFormatTypeExtension(formatType);

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(     \n" <<
                "       const uint offX,           \n" <<
                "       const uint offK,           \n" <<
                "       const uint N,              \n" <<
                "       __global const REAL* deltaVector,          \n" <<
                "       __global const REAL* X,    \n" <<
                "       __global const int* K,     \n" <<
                "       __global const REAL* Y,    \n" <<
                "       __global REAL* xBetaVector,      \n" <<
                "       __global REAL* expXBetaVector,   \n" <<
                "       __global REAL* denomPidVector,\n" <<
                "       __global const int* id,		\n" <<
				"		const uint stride,			\n" <<
				"		__global const int* cvIndices,	\n" <<
				"		const uint blockSize,		\n" <<
				"		__global const REAL* Offs) {   \n" <<
        "   uint task = get_global_id(0)%blockSize; \n" <<
		"	uint bufferIndex = get_global_id(0)/blockSize;	\n" <<
		"	uint vecOffset = stride*cvIndices[bufferIndex];	\n" <<
		"	REAL delta = deltaVector[cvIndices[bufferIndex]];	\n";
		//"	__local uint cvIndex, vecOffset;	\n" <<
		//"	__local REAL delta;					\n" <<
		//"	cvIndex = cvIndices[get_global_id(0)/blockSize];	\n" <<
		//"	delta = deltaVector[cvIndex];			\n" <<
		//"	vecOffset = stride*cvIndex;	\n";
        code << "   if (task < N) {      				\n";

        if (formatType == INDICATOR || formatType == SPARSE) {
            code << "   uint k = K[offK + task];         \n";
        } else { // DENSE, INTERCEPT
            code << "   uint k = task;            \n";
        }

        if (formatType == SPARSE || formatType == DENSE) {
            code << "   REAL inc = delta * X[offX + task]; \n";
        } else { // INDICATOR, INTERCEPT
            code << "   REAL inc = delta;           \n";
        }

        code << "       REAL xb = xBetaVector[vecOffset+k] + inc; 		\n" <<
                "       xBetaVector[vecOffset+k] = xb;                  \n";
        if (BaseModel::likelihoodHasDenominator) {
        	code << "REAL y = Y[k];\n" <<
        			"REAL offs = Offs[k];\n";
        	code << "REAL exb = " << BaseModelG::getOffsExpXBetaG() << ";\n";
               	code << "expXBetaVector[vecOffset+k] = exb;\n";
           		code << "denomPidVector[vecOffset+k] =" << BaseModelG::getDenomNullValueG() << "+ exb;\n";
        }
        code << "   } \n";
        code << "}    \n";

        return SourceCode(code.str(), name);
    }
    */

	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSync1UpdateXBetaKernel(FormatType formatType) {

        std::string name = "updateXBetaSync1" + getFormatTypeExtension(formatType);

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(     \n" <<
                "       __global const uint* offXVec,                  \n" <<
                "       __global const uint* offKVec,                  \n" <<
                "       __global const uint* NVec,                     \n" <<
                "       __global const REAL* deltaVector,          \n" <<
                "       __global const REAL* X,    \n" <<
                "       __global const int* K,     \n" <<
                "       __global const REAL* Y,    \n" <<
                "       __global REAL* xBetaVector,      \n" <<
                "       __global REAL* expXBetaVector,   \n" <<
                "       __global REAL* denomPidVector,\n" <<
				"		__global REAL* Offs,		\n" <<
				"		const uint stride,			\n" <<
				"		const uint indexWorkSize,		\n" <<
				"		const uint wgs,					\n" <<
				"		__global const int* indices,	\n" <<
				"		__global const int* cvIndices) {   \n";
        code << "   uint task = get_global_id(0)%indexWorkSize;  \n" <<
				"	__local uint bufferIndex, cvIndex, index, vecOffset, offX, offK, N;	\n" <<
				"	__local REAL delta;					\n" <<
				"	bufferIndex = get_group_id(0)/wgs;	\n" <<
				"	index = indices[bufferIndex];		\n" <<
				"	cvIndex = cvIndices[bufferIndex]; 	\n" <<
				"	vecOffset = stride*cvIndex;			\n" <<
				"	offX = offXVec[index];				\n" <<
				"	offK = offKVec[index];				\n" <<
				"	delta = deltaVector[bufferIndex];	\n" <<
				"	N = NVec[index];					\n";
        code << "   while (task < N) {      				\n";
        if (formatType == INDICATOR || formatType == SPARSE) {
            code << "   uint k = K[offK + task];         \n";
        } else { // DENSE, INTERCEPT
            code << "   uint k = task;            \n";
        }

        if (formatType == SPARSE || formatType == DENSE) {
            code << "   REAL inc = delta * X[offX + task]; \n";
        } else { // INDICATOR, INTERCEPT
            code << "   REAL inc = delta;           \n";
        }

        code << "       REAL xb = xBetaVector[vecOffset+k] + inc; 		\n" <<
                "       xBetaVector[vecOffset+k] = xb;                  \n";
        // hack for logistic only

        if (BaseModel::likelihoodHasDenominator) {
        	code << "REAL y = Y[k];\n" <<
        			"REAL offs = Offs[k];\n";
        	code << "REAL exb = " << BaseModelG::getOffsExpXBetaG() << ";\n";
               	code << "expXBetaVector[vecOffset+k] = exb;\n";
           		code << "denomPidVector[vecOffset+k] =" << BaseModelG::getDenomNullValueG() << "+ exb;\n";
        }

        code << "       task += indexWorkSize;			 				\n";
        code << "   } \n";
        code << "}    \n";

        return SourceCode(code.str(), name);
    }

/*
	template <class BaseModel, typename WeightType, class BaseModelG>
	    SourceCode
		GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForMMUpdateXBetaKernel(bool isNvidia) {

	        std::string name = "updateXBetaMM";

	        std::stringstream code;
	        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	        code << "__kernel void " << name << "(     \n" <<
	                "       __global const uint* offXVec,                  \n" <<
	                "       __global const uint* offKVec,                  \n" <<
					"		__global const uint* NVec,	\n" <<
	                "       __global const REAL* X,    \n" <<
	                "       __global const int* K,     \n" <<
	                "       __global const REAL* Y,    \n" <<
	                "       __global REAL* xBetaVector,      \n" <<
	                "       __global REAL* expXBetaVector,   \n" <<
	                "       __global REAL* denomPidVector,\n" <<
					"		__global REAL* Offs,		\n" <<
					"		const uint deltaStride,			\n" <<
					"		const uint cvIndexStride,		\n" <<
	                "       __global const REAL* deltaVector,          \n" <<
					"		__global const int* cvIndices,	\n" <<
					"		const uint persons) {   \n";
			code << "   uint lid = get_local_id(0); \n" <<
					"   uint task = lid;  \n" <<
	        		"	__local uint bufferIndex, index, cvIndex, vecOffset, offX, offK;		\n" <<
					"	bufferIndex = get_group_id(0)/persons;	\n" <<
					"	index = get_group_id(0)%persons;			\n" <<
					"	cvIndex = cvIndices[bufferIndex];	\n" <<
					"	vecOffset = cvIndexStride*cvIndex;	\n" <<
					"	offX = offXVec[index];				\n" <<
					"	offK = offKVec[index];				\n" <<
	                "   __local REAL scratch[TPB];  		\n" <<
					"	scratch[lid] = 0.0;					\n" <<
					"	REAL sum = 0.0;						\n" <<
					"	REAL N = NVec[index];				\n";
	        code << "   while (task < N) { 					\n" <<
	        		"	uint k = K[offK+task];				\n";


	                // Fused transformation-reduction
	                if (formatType == INDICATOR || formatType == SPARSE) {
	                    code << "       uint k = K[offK + task];         \n";
	                } else { // DENSE, INTERCEPT
	                    code << "       uint k = task;            \n";
	                }
	                if (formatType == SPARSE || formatType == DENSE) {
	                    code << "       const REAL x = X[offX + task]; \n";
	                } else { // INDICATOR, INTERCEPT
	                    // Do nothing
	                }


	                code << "sum += " << timesX("deltaVector[deltaStride*cvIndex+k]", formatType) << ";\n";


	        code << "	sum += deltaVector[deltaStride*cvIndex+k];	\n" <<
					"		task += TPB;					\n" <<
					"	}									\n";
	        code << "   scratch[lid] = sum; \n";
	        code << (isNvidia ? ReduceBody1<real,true>::body() : ReduceBody1<real,false>::body());
	        code << "   if (lid == 0) { \n" <<
					//"		if (index == 0 && cvIndices[bufferIndex] == 0) {	\n" <<
					//"			printf(\"writing to %d, value %f\", vecOffset+index, scratch[0]);		\n" <<
					//"		}									\n" <<
	        		"		REAL xb = xBetaVector[vecOffset+index] + scratch[0];	\n" <<
	                "   	xBetaVector[vecOffset+index] = xb; \n";
	        // hack for logistic only
	        if (BaseModel::likelihoodHasDenominator) {
	        	code << "	REAL y = Y[index];\n" <<
	        			"	REAL offs = Offs[index];\n";
	        	code << "	REAL exb = " << BaseModelG::getOffsExpXBetaG() << ";\n";
	            code << "	expXBetaVector[vecOffset+index] = exb;\n";
	            code << "	denomPidVector[vecOffset+index] =" << BaseModelG::getDenomNullValueG() << "+ exb;\n";
	        }
            code << "   } \n";

	        code << "}  \n"; // End of kernel

	        return SourceCode(code.str(), name);
	    }
	    */


	template <class BaseModel, typename WeightType, class BaseModelG>
		    SourceCode
			GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForMMUpdateXBetaKernel(bool isNvidia) {

		        std::string name = "updateXBetaMM";

		        std::stringstream code;
		        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

		        code << "__kernel void " << name << "(     \n" <<
		                "       __global const uint* offXVec,                  \n" <<
		                "       __global const uint* offKVec,                  \n" <<
						"		__global const uint* NVec,	\n" <<
		                "       __global const REAL* X,    \n" <<
		                "       __global const int* K,     \n" <<
		                "       __global const REAL* Y,    \n" <<
		                "       __global REAL* xBetaVector,      \n" <<
		                "       __global REAL* expXBetaVector,   \n" <<
		                "       __global REAL* denomPidVector,\n" <<
						"		__global const REAL* Offs,		\n" <<
						"		const uint cvIndexStride,		\n" <<
		                "       __global const REAL* deltaVector,	\n" <<
						"		const uint syncCVFolds) {   \n";

				code << "	uint lid0 = get_local_id(0);			\n" <<
						"	uint lid1 = get_local_id(1);			\n" <<
						//"	if (lid0==0 && lid1==0 && get_group_id(0) == 0) printf(\" running pid %d \", get_group_id(1));	\n" <<
						"	uint task1 = lid1;						\n" <<
						"	uint pid = get_group_id(1);				\n" <<
						"	uint cvIndex = get_group_id(0)*32+lid0;	\n" <<
						"	uint vecOffset = cvIndexStride*pid;		\n" <<
						"	REAL sum = 0.0;							\n" <<
						"	__local REAL scratch[32][8];			\n" <<
						"	__local uint offK, offX, N;				\n" <<
						"	offK = offKVec[pid];					\n" <<
						"	offX = offXVec[pid];					\n" <<
						"	N = NVec[pid];							\n" <<
						"	if (cvIndex < syncCVFolds) {			\n" <<
						"	while (task1 < N) {						\n" <<
		        		"		uint k = K[offK+task1];				\n";
		        code << "		sum += deltaVector[k*cvIndexStride+cvIndex];	\n" <<
						"		task1 += 8;							\n" <<
						"	}										\n";
		        code << "   scratch[lid0][lid1] = sum; 				\n";
				code << "   for(int j = 1; j < 8; j <<= 1) {          \n" <<
			            "       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
			            "       uint mask = (j << 1) - 1;               \n" <<
			            "       if ((lid1 & mask) == 0) {                \n" <<
			            "           scratch[lid0][lid1] += scratch[lid0][lid1 + j]; \n" <<
			            "       }                                       \n" <<
			            "   }                                           \n";
				code << "   if (lid1 == 0) { \n" <<
		        		"		REAL xb = xBetaVector[vecOffset+cvIndex] + scratch[lid0][0];	\n" <<
		                "   	xBetaVector[vecOffset+cvIndex] = xb; \n";
		        // hack for logistic only
		        if (BaseModel::likelihoodHasDenominator) {
		        	code << "	REAL y = Y[pid];\n" <<
		        			"	REAL offs = Offs[pid];\n";
		        	code << "	REAL exb = " << BaseModelG::getOffsExpXBetaG() << ";\n";
		            code << "	expXBetaVector[vecOffset+cvIndex] = exb;\n";
		            code << "	denomPidVector[vecOffset+cvIndex] =" << BaseModelG::getDenomNullValueG() << "+ exb;\n";
		        }
	            code << "   } \n";
	            code << "}	\n";

		        code << "}  \n"; // End of kernel

		        return SourceCode(code.str(), name);
		    }


	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForComputeRemainingStatisticsKernel() {

        std::string name = "computeRemainingStatistics";

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(     \n" <<
                "       const uint N,              \n" <<
				"		__global REAL* xBeta,	   \n" <<
                "       __global REAL* expXBeta,   \n" <<
                "       __global REAL* denominator,\n" <<
				"		__global REAL* Y,			\n" <<
				"		__global REAL* Offs,		\n" <<
                "       __global const int* id) {   \n" <<
                "   const uint task = get_global_id(0); \n";
        //code << "   const uint lid = get_local_id(0); \n" <<
        //        "   const uint loopSize = get_global_size(0); \n";
        // Local and thread storage
        code << "   if (task < N) {      				\n";
        if (BaseModel::exactCLR) {
#ifdef USE_LOG_SUM
#else
        	code << "const REAL xb = xBeta[task];	\n" <<
        			"REAL exb = exp(xb);			\n" <<
					"expXBeta[task] = exb;			\n";
#endif
        } else if (BaseModel::likelihoodHasDenominator) {
        	code << "const REAL y = Y[task];\n" <<
        			"const REAL xb = xBeta[task];\n" <<
					"const REAL offs = Offs[task];\n";
					//"const int k = task;";
        	code << "REAL exb = " << BaseModelG::getOffsExpXBetaG() << ";\n";
        	code << "expXBeta[task] = exb;\n";
    		code << "denominator[" << BaseModelG::getGroupG() << "] =" << BaseModelG::getDenomNullValueG() << "+ exb;\n";
        	//code << "denominator[task] = (REAL)1.0 + exb;\n";
        	//code << " 		REAL exb = exp(xBeta[task]);		\n" <<
        	//		"		expXBeta[task] = exb;		\n";
            //code << "       denominator[task] = (REAL)1.0 + exb; \n";// <<
        	//code << "expXBeta[k] = exp(xb); \n";
        	//code << "expXBeta[k] = exp(1); \n";

            // LOGISTIC MODEL ONLY
            //                     const real t = BaseModel::getOffsExpXBeta(offs, xBeta[k], y[k], k);
            //                     expXBeta[k] = t;
            //                     denominator[k] = static_cast<real>(1.0) + t;
            //             code << "    const REAL t = 0.0;               \n" <<
            //                     "   expXBeta[k] = exp(xBeta[k]);      \n" <<
            //                     "   denominator[k] = REAL(1.0) + tmp; \n";
        }

        code << "   } \n";
        code << "}    \n";
        return SourceCode(code.str(), name);
    }

	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncComputeRemainingStatisticsKernel() {

        std::string name = "computeRemainingStatistics";

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(     \n" <<
				"		__global REAL* xBetaVector,	   \n" <<
                "       __global REAL* expXBetaVector,   \n" <<
                "       __global REAL* denomPidVector,\n" <<
				"		__global REAL* Y,			\n" <<
				"		__global REAL* Offs,		\n" <<
                "       __global const int* pIdVector,		\n" <<
				"		const uint cvIndexStride,	\n" <<
				"		const uint size0,			\n" <<
				"		const uint syncCVFolds) {   \n" <<
				"	uint lid0 = get_local_id(0);	\n" <<
				"	uint cvIndex = get_group_id(0)*size0+lid0;	\n" <<
				"	uint task = get_group_id(1);	\n" <<
				"	if (cvIndex < syncCVFolds) {	\n" <<
				"		uint vecOffset = task*cvIndexStride;	\n" <<
				"		REAL y = Y[task];			\n" <<
				"		REAL xb = xBetaVector[vecOffset + cvIndex];	\n" <<
				"		REAL offs = Offs[task];		\n";
        code << "		REAL exb = " << BaseModelG::getOffsExpXBetaG() << ";\n";
        code << "		expXBetaVector[vecOffset + cvIndex] = exb;	\n";
		code << "		denomPidVector[vecOffset + cvIndex] =" << BaseModelG::getDenomNullValueG() << "+ exb;\n";
        code << "   } \n";
        code << "}    \n";
        return SourceCode(code.str(), name);
    }

/*
	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncComputeRemainingStatisticsKernel() {

        std::string name = "computeRemainingStatistics";

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(     \n" <<
                "       const uint N,              \n" <<
				"		__global const REAL* xBetaVector,	   \n" <<
                "       __global REAL* expXBetaVector,   \n" <<
                "       __global REAL* denomPidVector,\n" <<
				"		__global const REAL* Y,			\n" <<
				"		__global const REAL* Offs,		\n" <<
                "       __global const int* pIdVector,		\n" <<
				"		const uint stride,				\n" <<
				"		__global const int* cvIndices,	\n" <<
				"		__const uint blockSize) {   \n" <<
                "   uint task = get_global_id(0)%blockSize; \n" <<
				"	__local uint cvIndex, vecOffset;	\n" <<
        		"	cvIndex = cvIndices[get_global_id(0)/blockSize];	\n" <<
				"	vecOffset = stride*cvIndex;	\n";
        //code << "   const uint lid = get_local_id(0); \n" <<
        //        "   const uint loopSize = get_global_size(0); \n";
        // Local and thread storage
        code << "   if (task < N) {      				\n";
        if (BaseModel::likelihoodHasDenominator) {
        	code << "REAL y = Y[task];\n" <<
        			"REAL xb = xBetaVector[vecOffset+task];\n" <<
					"REAL offs = Offs[task];\n";
					//"const int k = task;";
        	code << "REAL exb = " << BaseModelG::getOffsExpXBetaG() << ";\n";
        	code << "expXBetaVector[vecOffset+task] = exb;\n";
    		code << "denomPidVector[vecOffset+" << BaseModelG::getGroupG() << "] =" << BaseModelG::getDenomNullValueG() << "+ exb;\n";
        }
        code << "   } \n";
        code << "}    \n";
        return SourceCode(code.str(), name);
    }
*/

	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
	GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForGetGradientObjective(bool useWeights, bool isNvidia) {
        std::string name;
	    if(useWeights) {
	        name = "getGradientObjectiveW";
	    } else {
	        name = "getGradientObjectiveN";
	    }

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(            \n" <<
                "       const uint N,                     \n" <<
                "       __global const REAL* Y,           \n" <<
                "       __global const REAL* xBeta,       \n" <<
                "       __global REAL* buffer,            \n" <<
                "       __global const REAL* weight) {    \n";    // TODO Make weight optional
        // Initialization
        code << "   const uint lid = get_local_id(0); \n" <<
                "   const uint loopSize = get_global_size(0); \n" <<
                "   uint task = get_global_id(0);  \n" <<
                    // Local and thread storage
                "   __local REAL scratch[TPB];  \n" <<
                "   REAL sum = 0.0; \n";
        code << "   while (task < N) { \n";
        if (useWeights) {
        	code << "       const REAL w = weight[task];\n";
        }
        code << "	const REAL xb = xBeta[task];     \n" <<
        		"	const REAL y = Y[task];			 \n";
        code << " sum += " << weight("y * xb", useWeights) << ";\n";
        // Bookkeeping
        code << "       task += loopSize; \n" <<
                "   } \n" <<
                    // Thread -> local
                "   scratch[lid] = sum; \n";
        code << (isNvidia ? ReduceBody1<real,true>::body() : ReduceBody1<real,false>::body());

        code << "   if (lid == 0) { \n" <<
                "       buffer[get_group_id(0)] = scratch[0]; \n" <<
                "   } \n";
        code << "}  \n"; // End of kernel
        return SourceCode(code.str(), name);
	}


	template <class BaseModel, typename WeightType, class BaseModelG>
	    SourceCode
		GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForGetGradientObjectiveSync(bool isNvidia) {
	        std::string name;
		        name = "getGradientObjectiveSync";

	        std::stringstream code;
	        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	        code << "__kernel void " << name << "(            \n" <<
	                "       const uint N,                     \n" <<
	                "       __global const REAL* Y,           \n" <<
	                "       __global const REAL* xBetaVector,       \n" <<
	                "       __global REAL* buffer,            \n" <<
					"		const uint cvIndexStride,					\n" <<
	                "       __global const REAL* weightVector,	\n" <<
					"		const uint size0,				\n" <<
					"		const uint syncCVFolds) {    \n";    // TODO Make weight optional
			code << "	uint lid0 = get_local_id(0);		\n" <<
					//"	uint task0 = get_group_id(0)*size0+lid0;	\n" <<
					"	uint task1 = get_group_id(1);		\n" <<
					"	uint cvIndex = get_group_id(0)*size0+lid0;	\n" <<
					"	REAL sum = 0.0;					\n" <<
					//"	if (cvIndex < syncCVFolds) {		\n" <<
					"	while (task1 < N) {					\n" <<
					"		uint vecOffset = task1*cvIndexStride;	\n" <<
					"		REAL w = weightVector[vecOffset+cvIndex];	\n" <<
					"		REAL y = Y[task1];				\n" <<
					"		REAL xb = xBetaVector[vecOffset+cvIndex];	\n" <<
					"		sum += w * y * xb;				\n" <<
					"		task1 += get_num_groups(1);		\n" <<
					"	} 									\n" <<
					"	buffer[cvIndex+cvIndexStride*get_group_id(1)] = sum;	\n" <<
				    //"   if (get_global_id(0) == 0) printf(\"inside kernel\");    \n" <<
					"	}									\n";
	        //code << "}  \n"; // End of kernel
	        return SourceCode(code.str(), name);
		}

	/*
	template <class BaseModel, typename WeightType, class BaseModelG>
	    SourceCode
		GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForGetGradientObjectiveSync(bool isNvidia) {
	        std::string name;
		        name = "getGradientObjectiveSync";


	        std::stringstream code;
	        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	        code << "__kernel void " << name << "(            \n" <<
	                "       const uint N,                     \n" <<
	                "       __global const REAL* Y,           \n" <<
	                "       __global const REAL* xBetaVector,       \n" <<
	                "       __global REAL* buffer,            \n" <<
					"		const uint workSize,				\n" <<
					"		const uint wgs,						\n" <<
					"		const uint stride,					\n" <<
	                "       __global const REAL* weightVector) {    \n";    // TODO Make weight optional
	        // Initialization
	        code << "   uint lid = get_local_id(0); \n" <<
	        		"	__local uint cvIndex, vecOffset;		\n" <<
					"	cvIndex = get_group_id(0)/wgs;			\n" <<
	                "   uint task = get_global_id(0)%workSize;  \n" <<
	                    // Local and thread storage
	                "   __local REAL scratch[TPB];  \n" <<
					"	vecOffset = stride * cvIndex;			\n" <<
	                "   REAL sum = 0.0; \n";
	        code << "   while (task < N) { \n";
	        //if (useWeights) {
	        code << "       REAL w = weightVector[vecOffset+task];\n";
	        //}
	        code << "		REAL xb = xBetaVector[vecOffset+task];     \n" <<
	        		"		REAL y = Y[task];			 \n" <<
	        	    "   	sum += w * y * xb;                 \n";
	        //code << " sum += " << weight("y * xb", useWeights) << ";\n";
	        // Bookkeeping
	        code << "       task += workSize; \n" <<
	                "   } \n" <<
	                    // Thread -> local
	                "   scratch[lid] = sum; \n";
	        code << (isNvidia ? ReduceBody1<real,true>::body() : ReduceBody1<real,false>::body());

	        code << "   if (lid == 0) { \n" <<
	                "       buffer[get_group_id(0)] = scratch[0]; \n" <<
	                "   } \n";
	        code << "}  \n"; // End of kernel
	        return SourceCode(code.str(), name);
		}
		*/


	template <class BaseModel, typename WeightType, class BaseModelG>
	SourceCode
	GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForGetLogLikelihood(bool useWeights, bool isNvidia) {
        std::string name;
	    if(useWeights) {
	        name = "getLogLikelihoodW";
	    } else {
	        name = "getLogLikelihoodN";
	    }

	    std::stringstream code;
	    code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	    code << "__kernel void " << name << "(            \n" <<
	    		"       const uint K,                     \n" <<
				"		 const uint N,					   \n" <<
				"       __global const REAL* Y,           \n" <<
				"       __global const REAL* xBeta,       \n" <<
				"		 __global const REAL* denominator,				\n" <<
				"		 __global const REAL* accDenominator,			\n" <<
				"       __global REAL* buffer,            \n" <<
				"       __global const REAL* Kweight,	   \n" <<
				"		 __global const REAL* Nweight) {    \n";    // TODO Make weight optional
	    // Initialization

	    code << "   const uint lid = get_local_id(0); \n" <<
	    		"   const uint loopSize = get_global_size(0); \n" <<
				"   uint task = get_global_id(0);  \n" <<
				// Local and thread storage
				"   __local REAL scratch[TPB];  \n" <<
				"   REAL sum = 0.0; \n";

		code << "   while (task < K) { \n";

	    if (useWeights) {
	    	code << "       const REAL wK = Kweight[task];\n";
	    }
	    code << "	const REAL xb = xBeta[task];     \n" <<
	    		"	const REAL y = Y[task];			 \n";
	    code << " sum += " << weightK(BaseModelG::logLikeNumeratorContribG(), useWeights) << ";\n";

	    if (BaseModel::likelihoodHasDenominator) {
	    	code << "if (task < N) {	\n";
	    	code << " const REAL wN = Nweight[task];\n";
	    	if (BaseModel::cumulativeGradientAndHessian) {
	    		code << "const REAL denom = accDenominator[task];\n";
	    	} else {
	    		code << "const REAL denom = denominator[task];\n";
	    	}
	    	code << "sum -= (REAL) " << BaseModelG::logLikeDenominatorContribG() << ";\n";
	    	code << "}\n";
	    }

	    // Bookkeeping
	    code << "       task += loopSize; \n" <<
	    		"   } \n";
				// Thread -> local
				code << "   scratch[lid] = sum; \n";
	    code << (isNvidia ? ReduceBody1<real,true>::body() : ReduceBody1<real,false>::body());

	    code << "   if (lid == 0) { \n" <<
	    		"       buffer[get_group_id(0)] = scratch[0]; \n" <<
				"   } \n";

	    code << "}  \n"; // End of kernel
	    return SourceCode(code.str(), name);
	}

	template <class BaseModel, typename WeightType, class BaseModelG>
	SourceCode
	GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForReduceCVBuffer() {
        std::string name = "reduceCVBuffer";

	    std::stringstream code;
	    code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	    code << "__kernel void " << name << "(            \n" <<
	    		"		__global REAL* buffer,				\n" <<
				"		__global REAL* bufferOut,			\n" <<
				"		const uint syncCVFolds,				\n" <<
				"		const uint cvIndexStride,			\n" <<
				"		const uint wgs) {    \n";    // TODO Make weight optional
	    // Initialization
	    code <<	"	uint cvIndex = get_group_id(0);			\n" <<
	    		"	__local REAL scratch[2][TPB];				\n" <<
				"	uint lid = get_local_id(0);				\n" <<
				"	if (lid < wgs) {						\n" <<
				"		scratch[0][lid] = buffer[lid*cvIndexStride+cvIndex];	\n" <<
				"		scratch[1][lid] = buffer[(lid+wgs)*cvIndexStride+cvIndex];	\n" <<
				"	}										\n" <<
	            "   for(int j = 1; j < wgs; j <<= 1) {          \n" <<
	            "       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
	            "       uint mask = (j << 1) - 1;               \n" <<
	            "       if ((lid & mask) == 0) {                \n" <<
	            "           scratch[0][lid] += scratch[0][lid + j]; \n" <<
	            "           scratch[1][lid] += scratch[1][lid + j]; \n" <<
	            "       }                                       \n" <<
	            "   }                                           \n" <<
				"	if (lid == 0) {							\n" <<
				"		bufferOut[cvIndex] = scratch[0][lid];	\n" <<
				"		bufferOut[cvIndex+syncCVFolds] = scratch[1][lid];	\n" <<
				"	}										\n" <<
				"	}										\n";
	    return SourceCode(code.str(), name);
	}


	template <class BaseModel, typename WeightType, class BaseModelG>
	SourceCode
	GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForProcessDeltaKernel(int priorType) {
        std::string name;
        if (priorType == 0) name = "ProcessDeltaKernelNone";
        if (priorType == 1) name = "ProcessDeltaKernelLaplace";
        if (priorType == 2) name = "ProcessDeltaKernelNormal";

	    std::stringstream code;
	    code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	    code << "__kernel void " << name << "(            \n" <<
	    		"		__global const REAL* buffer,				\n" <<
				"		__global REAL* deltaVector,			\n" <<
				"		const uint syncCVFolds,				\n" <<
				"		const uint cvIndexStride,			\n" <<
				"		const uint wgs,						\n" <<
				"		__global REAL* boundVector,				\n" <<
				"		__global const REAL* priorParams,			\n" <<
				"		__global const REAL* XjYVector,			\n" <<
				"		const uint J,						\n" <<
				"		const uint index,					\n" <<
				"		__global REAL* betaVector,			\n" <<
				"		__global uint* allZero,				\n" <<
				"		__global const int* doneVector) {    \n";    // TODO Make weight optional
	    // Initialization
	    code <<	"	__local uint cvIndex;					\n" <<
	    		"	cvIndex = get_group_id(0);				\n" <<
	    		//"	uint cvIndex = get_group_id(0);			\n" <<
	    		"	__local REAL scratch[2][TPB];				\n" <<
				"	uint lid = get_local_id(0);				\n" <<
				"	if (lid < wgs) {						\n" <<
				"		scratch[0][lid] = buffer[lid*cvIndexStride+cvIndex];	\n" <<
				"		scratch[1][lid] = buffer[(lid+wgs)*cvIndexStride+cvIndex];	\n" <<
				"	}										\n" <<
	            "   for(int j = 1; j < wgs; j <<= 1) {          \n" <<
	            "       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
	            "       uint mask = (j << 1) - 1;               \n" <<
	            "       if ((lid & mask) == 0) {                \n" <<
	            "           scratch[0][lid] += scratch[0][lid + j]; \n" <<
	            "           scratch[1][lid] += scratch[1][lid + j]; \n" <<
	            "       }                                       \n" <<
	            "   }                                           \n";

	    code << "	if (lid == 0) {							\n" <<
	    		"		__local uint offset;				\n" <<
				"		offset = index*cvIndexStride+cvIndex;			\n" <<
				"		__local REAL grad, hess, beta, delta;		\n" <<
				"		grad = scratch[0][lid] - XjYVector[offset];		\n" <<
				"		hess = scratch[1][lid];		\n" <<
    			"		beta = betaVector[offset];		\n";
	    		//"		uint offset = cvIndex*J+index;		\n" <<
				//"		REAL grad = scratch[0][lid] - XjYVector[offset];		\n" <<
				//"		REAL hess = scratch[1][lid];		\n" <<
    			//"		REAL beta = betaVector[offset];		\n";
	    if (priorType == 0) {
	    	code << " delta = -grad / hess;			\n";
	    }
	    if (priorType == 1) {
	    	code << "	REAL lambda = priorParams[index];	\n" <<
					"	REAL negupdate = - (grad - lambda) / hess; \n" <<
					"	REAL posupdate = - (grad + lambda) / hess; \n" <<
					"	if (beta == 0 ) {					\n" <<
					"		if (negupdate < 0) {			\n" <<
					"			delta = negupdate;			\n" <<
					"		} else if (posupdate > 0) {		\n" <<
					"			delta = posupdate;			\n" <<
					"		} else {						\n" <<
					"			delta = 0;					\n" <<
					"		}								\n" <<
					"	} else {							\n" <<
					"		if (beta < 0) {					\n" <<
					"			delta = negupdate;			\n" <<
					"			if (beta+delta > 0) delta = -beta;	\n" <<
					"		} else {						\n" <<
					"			delta = posupdate;			\n" <<
					"			if (beta+delta < 0) delta = -beta;	\n" <<
					"		}								\n" <<
					"	}									\n";
	    }
	    if (priorType == 2) {
	    	code << "	REAL var = priorParams[index];		\n" <<
					"	delta = - (grad + (beta / var)) / (hess + (1.0 / var));	\n";
	    }

	    code << "	delta = delta * doneVector[cvIndex];	\n" <<
	    		"	REAL bound = boundVector[offset];		\n" <<
	    		"	if (delta < -bound)	{					\n" <<
				"		delta = -bound;						\n" <<
				"	} else if (delta > bound) {				\n" <<
				"		delta = bound;						\n" <<
				"	}										\n" <<
				//"	REAL intermediate = 2;					\n" <<
				"	REAL intermediate = max(fabs(delta)*2, bound/2);	\n" <<
				"	intermediate = max(intermediate, 0.001);	\n" <<
				"	boundVector[offset] = intermediate;		\n" <<
				"	deltaVector[index*cvIndexStride+cvIndex] = delta;	\n" <<
				"	betaVector[offset] = delta + beta;		\n" <<
				"	if (delta != 0.0)	{						\n" <<
				"		allZero[0] = 0;						\n" <<
				"	}										\n" <<
				"	}										\n";

	    code << "	}										\n";
	    return SourceCode(code.str(), name);
	}

	/*
	 * dumber xbeta update for mm
	template <class BaseModel, typename WeightType, class BaseModelG>
	    SourceCode
	    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForMMUpdateXBetaKernel(bool isNvidia) {

	        std::string name = "updateXBetaMM";

	        std::stringstream code;
	        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	        code << "__kernel void " << name << "(     \n" <<
	                "       __global const uint* offXVec,                  \n" <<
	                "       __global const uint* offKVec,                  \n" <<
					"		__global const uint* NVec,	\n" <<
	                "       __global const REAL* X,    \n" <<
	                "       __global const int* K,     \n" <<
	                "       __global const REAL* Y,    \n" <<
	                "       __global REAL* xBetaVector,      \n" <<
	                "       __global REAL* expXBetaVector,   \n" <<
	                "       __global REAL* denomPidVector,\n" <<
					"		__global REAL* Offs,		\n" <<
					"		const uint stride,			\n" <<
	                "       __global const REAL* deltaVector,          \n" <<
					"		__global const int* indices,	\n" <<
					"		__global const int* cvIndices,	\n" <<
					"		__global const int* cvLengths,	\n" <<
					"		__global const int* cvOffsets,	\n" <<
					"		const uint persons) {   \n";
			code << "   uint lid = get_local_id(0); \n" <<
					"   uint task = lid;  \n" <<
	        		"	__local uint bufferIndex, index, vecOffset, offX, offK;		\n" <<
					"	bufferIndex = get_group_id(0)/persons;	\n" <<
					"	index = get_group_id(0)%persons;			\n" <<
					"	vecOffset = stride*cvIndices[bufferIndex];			\n" <<
					"	offX = offXVec[index];				\n" <<
					"	offK = offKVec[index];				\n" <<
	                "   __local REAL scratch[TPB];  		\n" <<
					"	scratch[lid] = 0.0;					\n" <<
					"	REAL sum = 0.0;						\n" <<
					"	REAL N = NVec[index];				\n" <<
					"	int searchStart = 0;				\n";
	        code << "   while (task < cvLengths[bufferIndex]) { \n" <<
					"		int currentIndex = indices[cvOffsets[bufferIndex]+task];	\n" <<
	        		"		int blah = searchStart;			\n" <<
	        		"		while (blah < N) {		\n" <<
					"			if (currentIndex == K[offK+blah]) {	\n" <<
					"				REAL inc = deltaVector[cvOffsets[bufferIndex]+task]; \n" << // * X[offX+blah];	\n" <<
					"				sum += inc;				\n" <<
					"				searchStart = blah+1;		\n" <<
					"				blah += N;				\n" <<
					//"				printf(\"inc\");		\n" <<
					"			} else if (currentIndex < K[offK+blah]) {\n" <<
					"				blah += N;				\n" <<
					"			} else if (currentIndex > K[offK+blah]) {	\n" <<
					"				searchStart = blah;		\n" <<
					"				blah++;					\n" <<
					"			}							\n" <<
					"		}								\n" <<
					"		task += TPB;					\n" <<
					"	}									\n";
	        code << "   scratch[lid] = sum; \n";
	        code << (isNvidia ? ReduceBody1<real,true>::body() : ReduceBody1<real,false>::body());
	        code << "   if (lid == 0) { \n" <<
					//"		if (index == 0 && cvIndices[bufferIndex] == 0) {	\n" <<
					//"			printf(\"writing to %d, value %f\", vecOffset+index, scratch[0]);		\n" <<
					//"		}									\n" <<
	        		"		REAL xb = xBetaVector[vecOffset+index] + scratch[0];	\n" <<
	                "   	xBetaVector[vecOffset+index] = xb; \n";
	        // hack for logistic only
	        if (BaseModel::likelihoodHasDenominator) {
	        	code << "	REAL y = Y[index];\n" <<
	        			"	REAL offs = Offs[index];\n";
	        	code << "	REAL exb = " << BaseModelG::getOffsExpXBetaG() << ";\n";
	            code << "	expXBetaVector[vecOffset+index] = exb;\n";
	            code << "	denomPidVector[vecOffset+index] =" << BaseModelG::getDenomNullValueG() << "+ exb;\n";
	        }
            code << "   } \n";

	        code << "}  \n"; // End of kernel

	        return SourceCode(code.str(), name);
	    }
*/


	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForDoItAllKernel(FormatType formatType, int priorType) {

		std::string name;
        if (priorType == 0) name = "doItAll" + getFormatTypeExtension(formatType) + "PriorNone";
        if (priorType == 1) name = "doItAll" + getFormatTypeExtension(formatType) + "PriorLaplace";
        if (priorType == 2) name = "doItAll" + getFormatTypeExtension(formatType) + "PriorNormal";

		std::stringstream code;
		code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

		code << "__kernel void " << name << "(            \n" <<
				"       __global const uint* offXVec,                  \n" <<
				"       __global const uint* offKVec,                  \n" <<
				"       __global const uint* NVec,                     \n" <<
				//"       const uint offX,                  \n" <<
				//"       const uint offK,                  \n" <<
				//"       const uint N,                     \n" <<
				"       __global const REAL* X,           \n" <<
				"       __global const int* K,            \n" <<
				//"       __global const REAL* Y,           \n" <<
				//"		__global const REAL* Offs,		  \n" <<
				"       __global REAL* xBetaVector,       \n" <<
				//"       __global REAL* expXBetaVector,    \n" <<
				//"       __global REAL* denomPidVector,	  \n" <<
				//"       __global const int* pIdVector,           \n" <<  // TODO Make id optional
				"       __global const REAL* weightVector,	\n" <<
				"		__global REAL* boundVector,				\n" <<
				"		__global const REAL* priorParams,			\n" <<
				"		__global const REAL* XjYVector,			\n" <<
				"		__global REAL* betaVector,			\n" <<
				"		__global const int* doneVector,		\n" <<
				"		const uint cvIndexStride,		\n" <<
				//"		const uint syncCVFolds,			\n" <<
				//"		const uint index)	{				\n";
				"		const uint indexStart,				\n" <<
				"		const uint length,				\n" <<
				"		__global const uint* indices,	\n" <<
				"		__global const uint* smStarts,	\n" <<
				"		__global const uint* smScales,	\n" <<
				"		__global const uint* smIndices) {   		 	\n";    // TODO Make weight optional
		// Initialization
		code << "	__local uint offK, offX, N, index, allZero;	\n";
		if (priorType == 1) {
			code << "__local REAL lambda;				\n";
		}
		if (priorType == 2) {
			code << "__local REAL var;				\n";
		}
		code << "	__local REAL grad[TPB1*TPB0];		\n" <<
				"	__local REAL hess[TPB1*TPB0];		\n" <<
				"	__local REAL deltaVec[TPB0];		\n" <<
				"	__local int localDone[TPB0];		\n" <<
				//"	__local int scratchInt[TPB0];		\n" <<
				"	__local REAL localXB[TPB1*3*TPB0];	\n" <<
				"	uint lid0 = get_local_id(0);			\n" <<
				"	uint smScale = smScales[get_group_id(0)];	\n" <<
				//"	uint myTPB0 = TPB0;					\n" <<
				//"	uint myTPB1 = TPB1;					\n" <<
				"	uint myTPB0 = TPB0 / smScale;		\n" <<
				"	uint myTPB1 = TPB1 * smScale;		\n" <<
				//"	if (get_global_id(0) == 0) printf(\"smScale %d tpb0 %d tpb1 %d \", smScale, myTPB0, myTPB1);	\n" <<
				"	uint mylid0 = lid0 % myTPB0;		\n" <<
				//"	uint mylid0 = lid0;					\n" <<
				//"	uint cvIndex = smStarts[get_group_id(0)] + mylid0;	\n" <<
				"	uint cvIndex = smIndices[smStarts[get_group_id(0)] + mylid0];	\n" <<
				//"	uint cvIndex = get_group_id(0)*TPB0 + lid0;	\n" <<
				"	uint lid1 = get_local_id(1);		\n" <<
				"	uint mylid1 = lid1 * smScale + lid0 / myTPB0;		\n" <<
				//"	uint mylid1 = lid1;					\n" <<

				"	if (mylid1 == 0) {					\n" <<
				"		int temp = doneVector[cvIndex];	\n" <<
				"		localDone[lid0] = temp;	\n" <<
				//"		//scratchInt[lid0] = temp;	\n" <<
				"	}									\n";
/*
		code << "	for(int j = 1; j < myTPB0; j <<= 1) {          	\n" <<
				"       barrier(CLK_LOCAL_MEM_FENCE);           	\n" <<
				"       uint mask = (j << 1) - 1;               	\n" <<
				"       if (mylid1==0 && (lid0 & mask) == 0) {                	\n" <<
				"           scratchInt[lid0] += scratchInt[lid0 + j]; 		\n" <<
				"       }                                       	\n" <<
				"	}									\n";

		code << "   barrier(CLK_LOCAL_MEM_FENCE);           \n";
		code << "	if (scratchInt[0] > 0) {			\n";
		*/
		code << "	for (int n = 0; n < length; n++) {	\n" <<
				"		index = indices[indexStart + n];	\n" <<
				"		offK = offKVec[index];			\n" <<
				"		offX = offXVec[index];			\n" <<
				"		N = NVec[index];				\n" <<
				"	uint task = mylid1;					\n" <<
				"	uint count = 0;						\n" <<
				"	REAL sum0 = 0.0;					\n" <<
				"	REAL sum1 = 0.0;					\n";
		code <<	"	while (task < N) {		\n";
		if (formatType == INDICATOR || formatType == SPARSE) {
			code << "  	uint k = K[offK + task];      	\n";
		} else { // DENSE, INTERCEPT
			code << "   uint k = task;           		\n";
		}
		if (formatType == SPARSE || formatType == DENSE) {
			code << "  	REAL x = X[offX + task]; \n";
		} else { // INDICATOR, INTERCEPT
			// Do nothing
		}
		code << "		uint vecOffset = k*cvIndexStride + cvIndex;	\n" <<
				"		REAL xb = xBetaVector[vecOffset];			\n" <<
				"		if (count < 3) localXB[(mylid1+myTPB1*count)*myTPB0 + mylid0] = xb;	\n" <<
				"		REAL exb = exp(xb);							\n" <<
				//"		REAL exb = expXBetaVector[vecOffset];	\n" <<
				"		REAL numer = " << timesX("exb", formatType) << ";\n" <<
				"		REAL denom = (REAL)1.0 + exb;				\n" <<
				//"		REAL denom = denomPidVector[vecOffset];		\n" <<
				"		REAL w = weightVector[vecOffset];\n";
		code << BaseModelG::incrementGradientAndHessianG(formatType, true);
		code << "       sum0 += gradient; \n" <<
				"       sum1 += hessian;  \n";
		code << "       task += myTPB1; \n" <<
				"		count += 1;		\n" <<
				"   } \n";

		code << "	grad[mylid1*myTPB0+mylid0] = sum0;	\n" <<
				"	hess[mylid1*myTPB0+mylid0] = sum1;	\n";

		code << "   for(int j = 1; j < myTPB1; j <<= 1) {          \n" <<
	            "       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
	            "       uint mask = (j << 1) - 1;               \n" <<
	            "       if ((mylid1 & mask) == 0) {                \n" <<
	            "           grad[mylid1*myTPB0+mylid0] += grad[(mylid1+j)*myTPB0+mylid0]; \n" <<
	            "           hess[mylid1*myTPB0+mylid0] += hess[(mylid1+j)*myTPB0+mylid0]; \n" <<
	            "       }                                       \n" <<
	            "   }                                         \n";

		code << "	if (mylid0 == 0 && mylid1 == 0) {			\n" <<
				"	allZero = 1;								\n";
		if (priorType == 1) {
			code << "lambda = priorParams[index];				\n";
		}
		if (priorType == 2) {
			code << "var = priorParams[index];				\n";
		}
		code << "	}										\n";
		code << "	barrier(CLK_LOCAL_MEM_FENCE);				\n";
		code << "	if (mylid1 == 0) {	\n" <<
				"		uint offset = cvIndexStride*index+cvIndex;		\n" <<
				"		REAL grad0 = grad[lid0];		\n" <<
				"		grad0 = grad0 - XjYVector[offset];	\n" <<
				"		REAL hess0 = hess[lid0];		\n" <<
    			"		REAL beta = betaVector[offset];		\n" <<
				"		REAL delta;							\n";

		if (priorType == 0) {
			code << " delta = -grad0 / hess0;			\n";
		}
		if (priorType == 1) {
			//code << "	REAL lambda = priorParams[index];	\n" <<
			code << "	REAL negupdate = - (grad0 - lambda) / hess0; \n" <<
					"	REAL posupdate = - (grad0 + lambda) / hess0; \n" <<
					"	if (beta == 0 ) {					\n" <<
					"		if (negupdate < 0) {			\n" <<
					"			delta = negupdate;			\n" <<
					"		} else if (posupdate > 0) {		\n" <<
					"			delta = posupdate;			\n" <<
					"		} else {						\n" <<
					"			delta = 0;					\n" <<
					"		}								\n" <<
					"	} else {							\n" <<
					"		if (beta < 0) {					\n" <<
					"			delta = negupdate;			\n" <<
					"			if (beta+delta > 0) delta = -beta;	\n" <<
					"		} else {						\n" <<
					"			delta = posupdate;			\n" <<
					"			if (beta+delta < 0) delta = -beta;	\n" <<
					"		}								\n" <<
					"	}									\n";
		}
		if (priorType == 2) {
			//code << "	REAL var = priorParams[index];		\n" <<
			code << "	delta = - (grad0 + (beta / var)) / (hess0 + (1.0 / var));	\n";
		}

		code << "	delta = delta * localDone[lid0];	\n" <<
				"	REAL bound = boundVector[offset];		\n" <<
				"	if (delta < -bound)	{					\n" <<
				"		delta = -bound;						\n" <<
				"	} else if (delta > bound) {				\n" <<
				"		delta = bound;						\n" <<
				"	}										\n" <<
				"	deltaVec[lid0] = delta;					\n" <<
				"	REAL intermediate = max(fabs(delta)*2, bound/2);	\n" <<
				"	intermediate = max(intermediate, 0.001);	\n" <<
				"	boundVector[offset] = intermediate;		\n" <<
				//"	betaVector[offset] = delta + beta;		\n" <<
				"	if (delta != 0) {						\n" <<
				"		betaVector[offset] = delta + beta;	\n" <<
				"		allZero = 0;						\n" <<
				"	}										\n";
/*
				"	if (delta == 0) {						\n" <<
				"		scratchInt[lid0] = 0;				\n" <<
				"	} else {								\n" <<
				"		scratchInt[lid0] = 1;				\n" <<
				"	}										\n" <<
				*/

		code << "	}										\n";
/*
		code << "	for(int j = 1; j < myTPB0; j <<= 1) {          	\n" <<
				"       barrier(CLK_LOCAL_MEM_FENCE);           	\n" <<
				"       uint mask = (j << 1) - 1;               	\n" <<
				"       if (mylid1==0 && (lid0 & mask) == 0) {                	\n" <<
				"           scratchInt[lid0] += scratchInt[lid0 + j]; 		\n" <<
				"       }                                       	\n" <<
				"	}									\n";
				*/
        code << "   barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
        		"	if (allZero == 0)		{				\n" <<
        		//"	if (scratchInt[0] > 0) {				\n" <<
        		"	REAL delta = deltaVec[mylid0];			\n" <<
				"	count = 0;							\n" <<
        		"	task = mylid1;						\n";

        code <<	"	while (task < N) {		\n";
        if (formatType == INDICATOR || formatType == SPARSE) {
        	code << "  	uint k = K[offK + task];      	\n";
        } else { // DENSE, INTERCEPT
        	code << "   uint k = task;           		\n";
        }
        if (formatType == SPARSE || formatType == DENSE) {
        	code << "   REAL inc = delta * X[offX + task]; \n";
        } else { // INDICATOR, INTERCEPT
        	code << "   REAL inc = delta;           	\n";
        }
        code << "		uint vecOffset = k*cvIndexStride + cvIndex;	\n" <<
        		"		REAL xb;						\n" <<
				"		if (count < 3) {				\n" <<
				"			xb = localXB[(mylid1+myTPB1*count)*myTPB0 + mylid0] + inc; \n" <<
				"		} else {						\n" <<
				"			xb = xBetaVector[vecOffset] + inc;	\n" <<
				"		}								\n" <<
        		//"		REAL xb = xBetaVector[vecOffset] + inc;	\n" <<
				"		xBetaVector[vecOffset] = xb;	\n";
        code << "		task += myTPB1;							\n" <<
        		"		count += 1;						\n";
        code << "} 										\n";

        code << "   barrier(CLK_GLOBAL_MEM_FENCE);           \n";
        code << "}	\n";

        code << "}	\n";
		//code << "}	\n";
		code << "}	\n";
		return SourceCode(code.str(), name);
	}

	template <class BaseModel, typename WeightType, class BaseModelG>
	    SourceCode
	    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForDoItAllSingleKernel(FormatType formatType, int priorType) {

			std::string name;
	        if (priorType == 0) name = "doItAllSingle" + getFormatTypeExtension(formatType) + "PriorNone";
	        if (priorType == 1) name = "doItAllSingle" + getFormatTypeExtension(formatType) + "PriorLaplace";
	        if (priorType == 2) name = "doItAllSingle" + getFormatTypeExtension(formatType) + "PriorNormal";

			std::stringstream code;
			code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

			code << "__kernel void " << name << "(            \n" <<
					"       __global const uint* offXVec,                  \n" <<
					"       __global const uint* offKVec,                  \n" <<
					"       __global const uint* NVec,                     \n" <<
					//"       const uint offX,                  \n" <<
					//"       const uint offK,                  \n" <<
					//"       const uint N,                     \n" <<
					"       __global const REAL* X,           \n" <<
					"       __global const int* K,            \n" <<
					//"       __global const REAL* Y,           \n" <<
					//"		__global const REAL* Offs,		  \n" <<
					"       __global REAL* xBetaVector,       \n" <<
					//"       __global REAL* expXBetaVector,    \n" <<
					//"       __global REAL* denomPidVector,	  \n" <<
					//"       __global const int* pIdVector,           \n" <<  // TODO Make id optional
					"       __global const REAL* weightVector,	\n" <<
					"		__global REAL* boundVector,				\n" <<
					"		__global const REAL* priorParams,			\n" <<
					"		__global const REAL* XjYVector,			\n" <<
					"		__global REAL* betaVector,			\n" <<
					"		__global const int* cvIndices,		\n" <<
					"		const uint cvIndexStride,		\n" <<
					//"		const uint syncCVFolds,			\n" <<
					//"		const uint index)	{				\n";
					"		const uint indexStart,				\n" <<
					"		const uint length,				\n" <<
					"		__global const uint* indices) {   		 	\n";    // TODO Make weight optional
			// Initialization
			code << "	__local uint offK, offX, N, index, cvIndex;	\n" <<
					//"	if (get_global_id(0)==0) printf(\"tpb = %d \", TPB);	\n" <<
					"	__local REAL scratch[2][TPB];			\n" <<
					"	__local REAL delta;					\n" <<
					"	__local REAL localXB[TPB*3];	\n" <<
					"	uint lid = get_local_id(0);			\n" <<
					"	cvIndex = cvIndices[get_group_id(0)];	\n";

			code << "	for (int n = 0; n < length; n++) {	\n" <<
					"		index = indices[indexStart + n];	\n" <<
					"		offK = offKVec[index];			\n" <<
					"		offX = offXVec[index];			\n" <<
					"		N = NVec[index];				\n" <<
					"		uint task = lid;				\n" <<
					"		uint count = 0;					\n" <<
					"		REAL sum0 = 0.0;				\n" <<
					"		REAL sum1 = 0.0;				\n";
			code <<	"		while (task < N) {				\n";
			if (formatType == INDICATOR || formatType == SPARSE) {
				code << "  		uint k = K[offK + task];      	\n";
			} else { // DENSE, INTERCEPT
				code << "   	uint k = task;           		\n";
			}
			if (formatType == SPARSE || formatType == DENSE) {
				code << "  		REAL x = X[offX + task]; \n";
			} else { // INDICATOR, INTERCEPT
				// Do nothing
			}
			code << "			uint vecOffset = k*cvIndexStride + cvIndex;	\n" <<
					"			REAL xb = xBetaVector[vecOffset];			\n" <<
					"			if (count < 3) localXB[count*TPB+lid] = xb;	\n" <<
					"			REAL exb = exp(xb);							\n" <<
					"			REAL numer = " << timesX("exb", formatType) << ";\n" <<
					"			REAL denom = (REAL)1.0 + exb;				\n" <<
					"			REAL w = weightVector[vecOffset];\n";
			code << BaseModelG::incrementGradientAndHessianG(formatType, true);
			code << "       	sum0 += gradient; \n" <<
					"       	sum1 += hessian;  \n";
			code << "       	task += TPB; \n" <<
					"			count += 1;		\n" <<
					"   	} \n";

			code << "		scratch[0][lid] = sum0;	\n" <<
					"		scratch[1][lid] = sum1;	\n";

	        code << ReduceBody2<real,false>::body();

			code << "		if (lid == 0) {	\n" <<
					"			uint offset = cvIndexStride*index+cvIndex;		\n" <<
					"			REAL grad0 = scratch[0][lid];			\n" <<
					"			grad0 = grad0 - XjYVector[offset];	\n" <<
					"			REAL hess0 = scratch[1][lid];			\n" <<
					"			REAL beta = betaVector[offset];		\n";

			if (priorType == 0) {
				code << " delta = -grad0 / hess0;			\n";
			}
			if (priorType == 1) {
				code << "		REAL lambda = priorParams[index];	\n" <<
						"		REAL negupdate = - (grad0 - lambda) / hess0; \n" <<
						"		REAL posupdate = - (grad0 + lambda) / hess0; \n" <<
						"		if (beta == 0 ) {					\n" <<
						"			if (negupdate < 0) {			\n" <<
						"				delta = negupdate;			\n" <<
						"			} else if (posupdate > 0) {		\n" <<
						"				delta = posupdate;			\n" <<
						"			} else {						\n" <<
						"				delta = 0;					\n" <<
						"			}								\n" <<
						"		} else {							\n" <<
						"			if (beta < 0) {					\n" <<
						"				delta = negupdate;			\n" <<
						"				if (beta+delta > 0) delta = -beta;	\n" <<
						"			} else {						\n" <<
						"				delta = posupdate;			\n" <<
						"				if (beta+delta < 0) delta = -beta;	\n" <<
						"			}								\n" <<
						"		}									\n";
			}
			if (priorType == 2) {
				code << "		REAL var = priorParams[index];		\n" <<
						"		delta = - (grad0 + (beta / var)) / (hess0 + (1.0 / var));	\n";
			}

			code << "			REAL bound = boundVector[offset];		\n" <<
					"			if (delta < -bound)	{					\n" <<
					"				delta = -bound;						\n" <<
					"			} else if (delta > bound) {				\n" <<
					"				delta = bound;						\n" <<
					"			}										\n" <<
					"			REAL intermediate = max(fabs(delta)*2, bound/2);	\n" <<
					"			intermediate = max(intermediate, 0.001);\n" <<
					"			boundVector[offset] = intermediate;		\n" <<
					"			if (delta != 0) betaVector[offset] = delta + beta;		\n" <<
					"		}										\n";
	        code << "   	barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
	        		"		if (delta != 0) {				\n" <<
					"			count = 0;							\n" <<
	        		"			task = lid;						\n";
	        code <<	"			while (task < N) {		\n";
	        if (formatType == INDICATOR || formatType == SPARSE) {
	        	code << "  			uint k = K[offK + task];      	\n";
	        } else { // DENSE, INTERCEPT
	        	code << "   		uint k = task;           		\n";
	        }
	        if (formatType == SPARSE || formatType == DENSE) {
	        	code << "   		REAL inc = delta * X[offX + task]; \n";
	        } else { // INDICATOR, INTERCEPT
	        	code << "   		REAL inc = delta;           	\n";
	        }
	        code << "				uint vecOffset = k*cvIndexStride + cvIndex;	\n" <<
	        		"				REAL xb;						\n" <<
					"				if (count < 3) {				\n" <<
					"					xb = localXB[count*TPB+lid] + inc; \n" <<
					"				} else {						\n" <<
					"					xb = xBetaVector[vecOffset] + inc;	\n" <<
					"				}								\n" <<
					"				xBetaVector[vecOffset] = xb;	\n";
	        code << "				task += TPB;					\n" <<
	        		"				count += 1;						\n";
	        code << "			} 									\n";
	        code << "   		barrier(CLK_GLOBAL_MEM_FENCE);      \n";
	        code << "		}	\n";
	        code << "	}	\n";
			code << "}	\n";
			return SourceCode(code.str(), name);
		}

	/*
	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForDoItAllKernel(FormatType formatType, int priorType) {

		std::string name;
        if (priorType == 0) name = "doItAll" + getFormatTypeExtension(formatType) + "PriorNone";
        if (priorType == 1) name = "doItAll" + getFormatTypeExtension(formatType) + "PriorLaplace";
        if (priorType == 2) name = "doItAll" + getFormatTypeExtension(formatType) + "PriorNormal";

		std::stringstream code;
		code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

		code << "__kernel void " << name << "(            \n" <<
				"       __global const uint* offXVec,                  \n" <<
				"       __global const uint* offKVec,                  \n" <<
				"       __global const uint* NVec,                     \n" <<
				//"       const uint offX,                  \n" <<
				//"       const uint offK,                  \n" <<
				//"       const uint N,                     \n" <<
				"       __global const REAL* X,           \n" <<
				"       __global const int* K,            \n" <<
				//"       __global const REAL* Y,           \n" <<
				//"		__global const REAL* Offs,		  \n" <<
				"       __global REAL* xBetaVector,       \n" <<
				//"       __global REAL* expXBetaVector,    \n" <<
				//"       __global REAL* denomPidVector,	  \n" <<
				//"       __global const int* pIdVector,           \n" <<  // TODO Make id optional
				"       __global const REAL* weightVector,	\n" <<
				"		__global REAL* boundVector,				\n" <<
				"		__global const REAL* priorParams,			\n" <<
				"		__global const REAL* XjYVector,			\n" <<
				"		__global REAL* betaVector,			\n" <<
				"		__global const int* cvIndices,		\n" <<
				"		const uint cvIndexStride,		\n" <<
				//"		const uint syncCVFolds,			\n" <<
				"		const uint J,						\n" <<
				//"		const uint index)	{				\n";
				"		const uint indexStart,				\n" <<
				"		const uint length,				\n" <<
				"		__global const uint* indices) {   		 	\n";    // TODO Make weight optional
		// Initialization
		code << "	__local uint cvIndex, remainder, loops, offK, offX, N, index;	\n" <<
				"	__local REAL scratch[2][TPB];		\n" <<
				//"	__local uint cvIndex, remainder, loops;	\n" <<
				"	cvIndex = cvIndices[get_group_id(0)];	\n" <<
				"	uint lid = get_local_id(0);			\n" <<
				"	for (int n = 0; n < length; n++) {	\n" <<
				"		index = indices[indexStart + n];	\n" <<
				"		offK = offKVec[index];			\n" <<
				"		offX = offXVec[index];			\n" <<
				"		N = NVec[index];				\n" <<
				"	uint task0;							\n" <<
				"	loops = N / TPB;					\n" <<
				"	remainder = N % TPB;				\n" <<
				"	if (lid < remainder) {				\n" <<
				"		task0 = lid*(loops+1);			\n" <<
				"	} else {							\n" <<
				"		task0 = remainder*(loops+1) + (lid-remainder)*loops;	\n" <<
				"	}									\n" <<
				"	uint task = task0;					\n" <<
				"	REAL sum0 = 0.0;					\n" <<
				"	REAL sum1 = 0.0;					\n";
		code <<	"	for (int i=0; i<loops; i++) {		\n";
		if (formatType == INDICATOR || formatType == SPARSE) {
			code << "  	uint k = K[offK + task];      	\n";
		} else { // DENSE, INTERCEPT
			code << "   uint k = task;           		\n";
		}
		if (formatType == SPARSE || formatType == DENSE) {
			code << "  	REAL x = X[offX + task]; \n";
		} else { // INDICATOR, INTERCEPT
			// Do nothing
		}
		code << "		uint vecOffset = cvIndex*cvIndexStride + k;	\n" <<
				"		REAL xb = xBetaVector[vecOffset];			\n" <<
				"		REAL exb = exp(xb);							\n" <<
				//"		REAL exb = expXBetaVector[vecOffset];	\n" <<
				"		REAL numer = " << timesX("exb", formatType) << ";\n" <<
				"		REAL denom = (REAL)1.0 + exb;				\n" <<
				//"		REAL denom = denomPidVector[vecOffset];		\n" <<
				"		REAL w = weightVector[vecOffset];\n";
		code << BaseModelG::incrementGradientAndHessianG(formatType, true);
		code << "       sum0 += gradient; \n" <<
				"       sum1 += hessian;  \n";
		code << "       task += 1; \n" <<
				"   } \n";

		code << "	if (lid < remainder)	{				\n";
		if (formatType == INDICATOR || formatType == SPARSE) {
			code << "  	uint k = K[offK + task];      	\n";
		} else { // DENSE, INTERCEPT
			code << "   uint k = task;           		\n";
		}
		if (formatType == SPARSE || formatType == DENSE) {
			code << "  	REAL x = X[offX + task]; \n";
		} else { // INDICATOR, INTERCEPT
			// Do nothing
		}
		code << "		uint vecOffset = cvIndex*cvIndexStride + k;	\n" <<
				"		REAL xb = xBetaVector[vecOffset];			\n" <<
				"		REAL exb = exp(xb);							\n" <<
				//"		REAL exb = expXBetaVector[vecOffset];	\n" <<
				"		REAL numer = " << timesX("exb", formatType) << ";\n" <<
				"		REAL denom = (REAL)1.0 + exb;				\n" <<
				//"		REAL denom = denomPidVector[vecOffset];		\n" <<
				"		REAL w = weightVector[vecOffset];\n";
		code << BaseModelG::incrementGradientAndHessianG(formatType, true);
		code << "       sum0 += gradient; \n" <<
				"       sum1 += hessian;  \n" <<
				"	}						\n";

		code << "	scratch[0][lid] = sum0;	\n" <<
				"	scratch[1][lid] = sum1;	\n";

        code << ReduceBody2<real,false>::body();


		code << "	__local REAL delta;						\n" <<
				"	if (lid == 0) {	\n" <<
				"		uint offset = cvIndex*J+index;		\n" <<
				"		REAL grad0 = scratch[0][0];		\n" <<
				"		grad0 = grad0 - XjYVector[offset];	\n" <<
				"		REAL hess0 = scratch[1][0];		\n" <<
    			"		REAL beta = betaVector[offset];		\n";

		if (priorType == 0) {
			code << " delta = -grad0 / hess0;			\n";
		}
		if (priorType == 1) {
			code << "	REAL lambda = priorParams[index];	\n" <<
					"	REAL negupdate = - (grad0 - lambda) / hess0; \n" <<
					"	REAL posupdate = - (grad0 + lambda) / hess0; \n" <<
					"	if (beta == 0 ) {					\n" <<
					"		if (negupdate < 0) {			\n" <<
					"			delta = negupdate;			\n" <<
					"		} else if (posupdate > 0) {		\n" <<
					"			delta = posupdate;			\n" <<
					"		} else {						\n" <<
					"			delta = 0;					\n" <<
					"		}								\n" <<
					"	} else {							\n" <<
					"		if (beta < 0) {					\n" <<
					"			delta = negupdate;			\n" <<
					"			if (beta+delta > 0) delta = -beta;	\n" <<
					"		} else {						\n" <<
					"			delta = posupdate;			\n" <<
					"			if (beta+delta < 0) delta = -beta;	\n" <<
					"		}								\n" <<
					"	}									\n";
		}
		if (priorType == 2) {
			code << "	REAL var = priorParams[index];		\n" <<
					"	delta = - (grad0 + (beta / var)) / (hess0 + (1.0 / var));	\n";
		}

		code << "	REAL bound = boundVector[offset];		\n" <<
				"	if (delta < -bound)	{					\n" <<
				"		delta = -bound;						\n" <<
				"	} else if (delta > bound) {				\n" <<
				"		delta = bound;						\n" <<
				"	}										\n" <<
				"	REAL intermediate = max(fabs(delta)*2, bound/2);	\n" <<
				"	intermediate = max(intermediate, 0.001);	\n" <<
				"	boundVector[offset] = intermediate;		\n" <<
				"	betaVector[offset] = delta + beta;		\n" <<
				"	}										\n";

        code << "   barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
        		"	if (delta != 0) {					\n" <<
        		"	task = task0;						\n";
        code <<	"	for (int i=0; i<loops; i++) {		\n";
        if (formatType == INDICATOR || formatType == SPARSE) {
        	code << "  	uint k = K[offK + task];      	\n";
        } else { // DENSE, INTERCEPT
        	code << "   uint k = task;           		\n";
        }
        if (formatType == SPARSE || formatType == DENSE) {
        	code << "   REAL inc = delta * X[offX + task]; \n";
        } else { // INDICATOR, INTERCEPT
        	code << "   REAL inc = delta;           	\n";
        }
        code << "		uint vecOffset = cvIndex*cvIndexStride + k;	\n" <<
        		"		REAL xb = xBetaVector[vecOffset] + inc;	\n" <<
				"		xBetaVector[vecOffset] = xb;	\n";
        code << "	task += 1;							\n";
        code << "} 										\n";

        code <<	"if (lid < remainder) {		\n";
        if (formatType == INDICATOR || formatType == SPARSE) {
        	code << "  	uint k = K[offK + task];      	\n";
        } else { // DENSE, INTERCEPT
        	code << "   uint k = task;           		\n";
        }
        if (formatType == SPARSE || formatType == DENSE) {
        	code << "   REAL inc = delta * X[offX + task]; \n";
        } else { // INDICATOR, INTERCEPT
        	code << "   REAL inc = delta;           	\n";
        }
        code << "		uint vecOffset = cvIndex*cvIndexStride + k;	\n" <<
        		"		REAL xb = xBetaVector[vecOffset] + inc;	\n" <<
				"		xBetaVector[vecOffset] = xb;	\n";
        code << "} 										\n";
        code << "}    	\n";
        code << "   barrier(CLK_GLOBAL_MEM_FENCE);           \n";
		code << "}  \n"; // End of kernel
		code << "}	\n";
		return SourceCode(code.str(), name);
	}
	*/

	template <class BaseModel, typename WeightType, class BaseModelG>
	    SourceCode
	    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForExactCLRDoItAllKernel(FormatType formatType, int priorType) {
			std::string name;
			        if (priorType == 0) name = "doItAll" + getFormatTypeExtension(formatType) + "PriorNone";
			        if (priorType == 1) name = "doItAll" + getFormatTypeExtension(formatType) + "PriorLaplace";
			        if (priorType == 2) name = "doItAll" + getFormatTypeExtension(formatType) + "PriorNormal";

			std::stringstream code;
				code << "REAL log_sum(REAL x, REAL y) {										\n" <<
						"	if (isinf(x)) return y;											\n" <<
						"	if (isinf(y)) return x;											\n" <<
						"	if (x > y) {											\n" <<
						"		return x + log(1 + exp(y-x));								\n" <<
						"	} else {														\n" <<
						"		return y + log(1 + exp(x-y));								\n" <<
						"	}																\n" <<
						//"	REAL z = max(x,y);												\n" <<
						//"	return z + log(exp(x-z) + exp(y-z));							\n" <<
						"}																	\n";

			code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

			code << "__kernel void " << name << "(            \n" <<
					"       __global const uint* offXVec,                  \n" <<
					"       __global const uint* offKVec,                  \n" <<
					"       __global const uint* NVec,                     \n" <<
					//"       const uint offX,                  \n" <<
					//"       const uint offK,                  \n" <<
					//"       const uint N,                     \n" <<
					"       __global const REAL* X,           \n" <<
					"       __global const int* K,            \n" <<
					//"       __global const REAL* Y,           \n" <<
					//"		__global const REAL* Offs,		  \n" <<
					"       __global REAL* xBetaVector,       \n" <<
					//"       __global REAL* expXBetaVector,    \n" <<
					//"       __global REAL* denomPidVector,	  \n" <<
					//"       __global const int* pIdVector,           \n" <<  // TODO Make id optional
					"       __global const REAL* weightVector,	\n" <<
					"		__global REAL* boundVector,				\n" <<
					"		__global const REAL* priorParams,			\n" <<
					"		__global const REAL* XjYVector,			\n" <<
					"		__global REAL* betaVector,			\n" <<
					"		__global const int* doneVector,		\n" <<
					"		const uint cvIndexStride,		\n" <<
					//"		const uint syncCVFolds,			\n" <<
					//"		const uint index)	{				\n";
					"		const uint indexStart,				\n" <<
					"		const uint length,				\n" <<
					"		__global const uint* indices,	\n" <<
					"		__global const uint* smStarts,	\n" <<
					"		__global const uint* smScales,	\n" <<
					"		__global const uint* smIndices,	\n" <<
					"		__global const uint* NtoK,		\n" <<
					"		__global const REAL* casesVec,	\n" <<
					"		const uint persons,				\n" <<
					"		const uint totalStrata) {   		 	\n";    // TODO Make weight optional

			//"	__global REAL* firstRow,				\n" <<
			//"	__global const uint* KStrata,			\n" <<

			// Initialization
			code << "	__local uint offK, offX, N, index, allZero;	\n";
			if (priorType == 1) {
				code << "__local REAL lambda;				\n";
			}
			if (priorType == 2) {
				code << "__local REAL var;				\n";
			}
			code << "	__local REAL grad[TPB0];		\n" <<
					"	__local REAL hess[TPB0];		\n" <<
					"	__local REAL deltaVec[TPB0];		\n" <<
					"	__local int localDone[TPB0];		\n" <<
					"	__local REAL B0[2][TPB0*TPB1];			\n" <<
					"	__local REAL B1[2][TPB0*TPB1];			\n" <<
					"	__local REAL B2[2][TPB0*TPB1];			\n" <<
					"	__local REAL localWeights[TPB0];			\n" <<
					//"	__local int scratchInt[TPB0];		\n" <<
					"	uint lid0 = get_local_id(0);			\n" <<
					"	uint lid1 = get_local_id(1);			\n" <<
					"	uint smScale = smScales[get_group_id(0)];	\n" <<
					"	uint myTPB0 = TPB0 / smScale;		\n" <<
					"	uint myTPB1 = TPB1 * smScale;		\n" <<
					"	uint mylid0 = lid0 % myTPB0;		\n" <<
					"	uint mylid1 = lid1 * smScale + lid0 / myTPB0;		\n" <<
					"	uint mylid = mylid1*myTPB0+mylid0;	\n" <<
					"	uint cvIndex = smIndices[smStarts[get_group_id(0)] + lid0];	\n" <<

					"	if (mylid1 == 0) {					\n" <<
					"		int temp = doneVector[cvIndex];	\n" <<
					"		localDone[lid0] = temp;	\n" <<
					"	}									\n";

			code << "	for (int n = 0; n < length; n++) {	\n" <<
					"		index = indices[indexStart + n];	\n" <<
					"		offK = offKVec[index];			\n" <<
					"		offX = offXVec[index];			\n" <<
					"		N = NVec[index];				\n";

			code << "	for (int stratum = 0; stratum < totalStrata; stratum++) {	\n" <<
					"		int stratumStart = NtoK[stratum];	\n" <<
					"		uint vecOffset = stratumStart*cvIndexStride + cvIndex;	\n" <<
					"		int total = NtoK[stratum+1] - stratumStart;	\n" <<
					"		int cases = casesVec[stratum];		\n" <<
					"		int controls = total - cases;		\n" <<
					"		int offKStrata = index*totalStrata + stratum;	\n" <<
					"		if (mylid1 == 0) {					\n" <<
					"			REAL temp = weightVector[vecOffset];	\n" <<
					"			localWeights[mylid0] = temp;		\n" <<
					"		}									\n" <<
					"		barrier(CLK_LOCAL_MEM_FENCE);		\n";
#ifdef USE_LOG_SUM
			code << "		B0[0][mylid] = -INFINITY;					\n" <<
					"		B0[1][mylid] = -INFINITY;					\n" <<
					"		B1[0][mylid] = -INFINITY;					\n" <<
					"		B1[1][mylid] = -INFINITY;					\n" <<
					"		B2[0][mylid] = -INFINITY;					\n" <<
					"		B2[1][mylid] = -INFINITY;					\n" <<
					"		if (mylid1 == 0) {							\n" <<
					"			B0[0][mylid0] = 0;						\n" <<
					"			B0[1][mylid0] = 0;						\n" <<
					"		}										\n" <<
					"		const REAL logTwo = log((REAL)2.0);		\n";
#else
			code << "		B0[0][mylid] = 0;							\n" <<
					"		B0[1][mylid] = 0;							\n" <<
					"		B1[0][mylid] = 0;							\n" <<
					"		B1[1][mylid] = 0;							\n" <<
					"		B2[0][mylid] = 0;							\n" <<
					"		B2[1][mylid] = 0;							\n" <<
					"		if (mylid1 == 0) {							\n" <<
					"			B0[0][mylid0] = 1;						\n" <<
					"			B0[1][mylid0] = 1;						\n" <<
					"		}										\n";
#endif
			//"	uint current = 0;						\n";

			code << "		uint loops;								\n" <<
					"		loops = cases / (myTPB1 - 1);				\n" <<
					"		if (cases % (myTPB1 - 1) > 0) {			\n" <<
					"			loops++;							\n" <<
					"		}										\n";

			// if loops == 1
			//code << "		if (loops == 1) {							\n" <<
			code << "			uint current = 0;						\n";

			if (formatType == INDICATOR || formatType == SPARSE) {
				code << "	__local uint currentKIndex, currentK;	\n" <<
						"	if (mylid0 == 0 && mylid1 == 0) {							\n" <<
						"		currentKIndex = KStrata[offKStrata];					\n" <<
						"		if (currentKIndex == -1) {			\n" <<
						"			currentK = -1;					\n" <<
						"		} else {							\n" <<
						"			currentK = K[offK+currentKIndex];	\n" <<
						"		}									\n" <<
						"	}										\n" <<
						"	barrier(CLK_LOCAL_MEM_FENCE);			\n";
			}

			if (formatType == INTERCEPT) {
				code << "REAL x;						\n";
			} else {
				code << "__local REAL x;				\n";
			}

			code << "	for (int col = 0; col < total; col++) {	\n" <<
#ifdef USE_LOG_SUM
					"		REAL U = xBetaVector[vecOffset + col * cvIndexStride];	\n";
#else
					"		REAL U = exp(xBetaVector[vecOffset + col * cvIndexStride]);	\n";
#endif

			if (formatType == DENSE) {
				code << "	if (mylid0 == 0 && mylid1 == 0) {						\n" <<
						"		x = X[offX+stratumStart+col];			\n" <<
						"	}									\n" <<
						"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
			}

			if (formatType == INTERCEPT) {
#ifdef USE_LOG_SUM
				code << "	x = 0;					\n";
#else
				code << "	x = 1;					\n";
#endif
			}

			if (formatType == INDICATOR) {
				code << "	if (mylid0 == 0 && mylid1 == 0) {						\n" <<
						"		if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
#ifdef USE_LOG_SUM
						"	x = -INFINITY;								\n" <<
#else
						"	x = 0;								\n" <<
#endif
						"		} else {	\n" <<
#ifdef USE_LOG_SUM
						"	x = 0;								\n" <<
#else
						"	x = 1;								\n" <<
#endif
						"			currentKIndex++;			\n" <<
						"			currentK = K[offK + currentKIndex];	\n" <<
						"		}								\n" <<
						"	}									\n" <<
						"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
			}

			if (formatType == SPARSE) {
				code << "	if (mylid0 == 0 && mylid1 == 0) {						\n" <<
						"		if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
#ifdef USE_LOG_SUM
						"	x = -INFINITY;								\n" <<
#else
						"	x = 0;								\n" <<
#endif
						"		} else {						\n" <<
						"			x = X[offX+currentKIndex];	\n" <<
						"			currentKIndex++;			\n" <<
						"			currentK = K[offK + currentKIndex];	\n" <<
						"		}								\n" <<
						"	}									\n" <<
						"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
			}

			code << "		if (mylid1 > 0 && mylid1 <= cases) {						\n" <<
#ifdef USE_LOG_SUM
					//"			x = log(x);										\n" <<
					"			B0[current][mylid] = log_sum(				 B0[1-current][mylid], U+B0[1-current][mylid-myTPB0]);	\n" <<
					"			B1[current][mylid] = log_sum(log_sum(		 B1[1-current][mylid], U+B1[1-current][mylid-myTPB0]), x + U + B0[1-current][mylid-myTPB0]);	\n" <<
					"			B2[current][mylid] = log_sum(log_sum(log_sum(B2[1-current][mylid], U+B2[1-current][mylid-myTPB0]), x + U + B0[1-current][mylid-myTPB0]), logTwo + x + U + B1[1-current][mylid-myTPB0]);	\n" <<

#else
					"			B0[current][mylid] = B0[1-current][mylid] + U*B0[1-current][mylid-myTPB0];	\n" <<
					"			B1[current][mylid] = B1[1-current][mylid] + U*B1[1-current][mylid-myTPB0] + x*U*B0[1-current][[mylid-myTPB0];	\n" <<
					"			B2[current][mylid] = B2[1-current][mylid] + U*B2[1-current][mylid-myTPB0] + x*U*B0[1-current][[mylid-myTPB0] + 2*x*U*B1[1-current][mylid-myTPB0];	\n" <<
#endif
					"		}									\n" <<
					"		current = 1 - current;				\n" <<
					"		barrier(CLK_LOCAL_MEM_FENCE);		\n" <<
					"	}										\n";


			code << "	if (mylid1 == cases) {							\n" <<
					"		if (localWeights[mylid0] != 0) {			\n" <<
					"		REAL value0 = B0[1-current][mylid]*localWeights[mylid0];	\n" <<
					"		REAL value1 = B1[1-current][mylid]*localWeights[mylid0];	\n" <<
					"		REAL value2 = B2[1-current][mylid]*localWeights[mylid0];	\n" <<
#ifdef USE_LOG_SUM
					"		grad[mylid0] += -exp(value1 - value0);			\n" <<
					"		hess[mylid0] += exp(2*(value1-value0)) - exp(value2 - value0);	\n" <<
#else
					"		grad[mylid0] += -value1/value0;					\n" <<
					"		hess[mylid0] += value1*value1/value0/value0 - value2/value0;		\n" <<
#endif
					"		}									\n" <<
					"	}										\n";

			code << "	barrier(CLK_GLOBAL_MEM_FENCE);			\n";
			code << "	barrier(CLK_LOCAL_MEM_FENCE);			\n";

			code << "}											\n";


			code << "	if (mylid0 == 0 && mylid1 == 0) {			\n" <<
					"	allZero = 1;								\n";
			if (priorType == 1) {
				code << "lambda = priorParams[index];				\n";
			}
			if (priorType == 2) {
				code << "var = priorParams[index];				\n";
			}
			code << "	}										\n";
			code << "	barrier(CLK_LOCAL_MEM_FENCE);				\n";
			code << "	if (mylid1 == 0) {	\n" <<
					"		uint offset = cvIndexStride*index+cvIndex;		\n" <<
					"		REAL grad0 = grad[mylid0];		\n" <<
					"		grad0 = grad0 - XjYVector[offset];	\n" <<
					"		REAL hess0 = hess[mylid0];		\n" <<
	    			"		REAL beta = betaVector[offset];		\n" <<
					"		REAL delta;							\n";

			if (priorType == 0) {
				code << " delta = -grad0 / hess0;			\n";
			}
			if (priorType == 1) {
				//code << "	REAL lambda = priorParams[index];	\n" <<
				code << "	REAL negupdate = - (grad0 - lambda) / hess0; \n" <<
						"	REAL posupdate = - (grad0 + lambda) / hess0; \n" <<
						"	if (beta == 0 ) {					\n" <<
						"		if (negupdate < 0) {			\n" <<
						"			delta = negupdate;			\n" <<
						"		} else if (posupdate > 0) {		\n" <<
						"			delta = posupdate;			\n" <<
						"		} else {						\n" <<
						"			delta = 0;					\n" <<
						"		}								\n" <<
						"	} else {							\n" <<
						"		if (beta < 0) {					\n" <<
						"			delta = negupdate;			\n" <<
						"			if (beta+delta > 0) delta = -beta;	\n" <<
						"		} else {						\n" <<
						"			delta = posupdate;			\n" <<
						"			if (beta+delta < 0) delta = -beta;	\n" <<
						"		}								\n" <<
						"	}									\n";
			}
			if (priorType == 2) {
				//code << "	REAL var = priorParams[index];		\n" <<
				code << "	delta = - (grad0 + (beta / var)) / (hess0 + (1.0 / var));	\n";
			}

			code << "	delta = delta * localDone[lid0];	\n" <<
					"	REAL bound = boundVector[offset];		\n" <<
					"	if (delta < -bound)	{					\n" <<
					"		delta = -bound;						\n" <<
					"	} else if (delta > bound) {				\n" <<
					"		delta = bound;						\n" <<
					"	}										\n" <<
					"	deltaVec[lid0] = delta;					\n" <<
					"	REAL intermediate = max(fabs(delta)*2, bound/2);	\n" <<
					"	intermediate = max(intermediate, 0.001);	\n" <<
					"	boundVector[offset] = intermediate;		\n" <<
					//"	betaVector[offset] = delta + beta;		\n" <<
					"	if (delta != 0) {						\n" <<
					"		betaVector[offset] = delta + beta;	\n" <<
					"		allZero = 0;						\n" <<
					"	}										\n";

			code << "	}										\n";

	        code << "   barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
	        		"	if (allZero == 0)		{				\n" <<
	        		//"	if (scratchInt[0] > 0) {				\n" <<
	        		"	REAL delta = deltaVec[mylid0];			\n" <<
	        		"	uint task = mylid1;						\n";

	        code <<	"	while (task < N) {		\n";
	        if (formatType == INDICATOR || formatType == SPARSE) {
	        	code << "  	uint k = K[offK + task];      	\n";
	        } else { // DENSE, INTERCEPT
	        	code << "   uint k = task;           		\n";
	        }
	        if (formatType == SPARSE || formatType == DENSE) {
	        	code << "   REAL inc = delta * X[offX + task]; \n";
	        } else { // INDICATOR, INTERCEPT
	        	code << "   REAL inc = delta;           	\n";
	        }
	        code << "		uint vecOffset = k*cvIndexStride + cvIndex;	\n" <<
	        		"		REAL xb;						\n" <<
					"		xb = xBetaVector[vecOffset] + inc;	\n" <<
	        		//"		REAL xb = xBetaVector[vecOffset] + inc;	\n" <<
					"		xBetaVector[vecOffset] = xb;	\n";
	        code << "		task += myTPB1;							\n";
	        code << "} 										\n";

	        code << "   barrier(CLK_GLOBAL_MEM_FENCE);           \n";
	        code << "}	\n";


	        code << "}	\n";
			//code << "}	\n";
			code << "}	\n";
			return SourceCode(code.str(), name);
		}

	template <class BaseModel, typename WeightType, class BaseModelG>
		    SourceCode
		    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForExactCLRDoItAllSingleKernel(FormatType formatType, int priorType) {
				std::string name;
		        if (priorType == 0) name = "doItAllSingle" + getFormatTypeExtension(formatType) + "PriorNone";
		        if (priorType == 1) name = "doItAllSingle" + getFormatTypeExtension(formatType) + "PriorLaplace";
		        if (priorType == 2) name = "doItAllSingle" + getFormatTypeExtension(formatType) + "PriorNormal";

				std::stringstream code;
					code << "REAL log_sum(REAL x, REAL y) {										\n" <<
							"	if (isinf(x)) return y;											\n" <<
							"	if (isinf(y)) return x;											\n" <<
							"	if (x > y) {											\n" <<
							"		return x + log(1 + exp(y-x));								\n" <<
							"	} else {														\n" <<
							"		return y + log(1 + exp(x-y));								\n" <<
							"	}																\n" <<
							//"	REAL z = max(x,y);												\n" <<
							//"	return z + log(exp(x-z) + exp(y-z));							\n" <<
							"}																	\n";

				code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

				code << "__kernel void " << name << "(            \n" <<
						"       __global const uint* offXVec,                  \n" <<
						"       __global const uint* offKVec,                  \n" <<
						"       __global const uint* NVec,                     \n" <<
						//"       const uint offX,                  \n" <<
						//"       const uint offK,                  \n" <<
						//"       const uint N,                     \n" <<
						"       __global const REAL* X,           \n" <<
						"       __global const int* K,            \n" <<
						//"       __global const REAL* Y,           \n" <<
						//"		__global const REAL* Offs,		  \n" <<
						"       __global REAL* xBetaVector,       \n" <<
						//"       __global REAL* expXBetaVector,    \n" <<
						//"       __global REAL* denomPidVector,	  \n" <<
						//"       __global const int* pIdVector,           \n" <<  // TODO Make id optional
						"       __global const REAL* weightVector,	\n" <<
						"		__global REAL* boundVector,				\n" <<
						"		__global const REAL* priorParams,			\n" <<
						"		__global const REAL* XjYVector,			\n" <<
						"		__global REAL* betaVector,			\n" <<
						"		__global const int* cvIndices,		\n" <<
						"		const uint cvIndexStride,		\n" <<
						//"		const uint syncCVFolds,			\n" <<
						//"		const uint index)	{				\n";
						"		const uint indexStart,				\n" <<
						"		const uint length,				\n" <<
						"		__global const uint* indices,	\n" <<
						"		__global const uint* NtoK,		\n" <<
						"		__global const REAL* casesVec,	\n" <<
						"		const uint persons,				\n" <<
						"		const uint totalStrata) {   		 	\n";    // TODO Make weight optional

				// Initialization
				code << "	__local uint offK, offX, N, index, cvIndex;	\n";
				if (priorType == 1) {
					code << "__local REAL lambda;				\n";
				}
				if (priorType == 2) {
					code << "__local REAL var;				\n";
				}
				code << "	__local REAL grad, hess, delta, weight;		\n" <<
						"	__local REAL B0[2][TPB];			\n" <<
						"	__local REAL B1[2][TPB];			\n" <<
						"	__local REAL B2[2][TPB];			\n" <<
						"	uint lid = get_local_id(0);			\n" <<
						"	grad = 0;							\n" <<
						"	hess = 0;							\n" <<

						"	cvIndex = cvIndices[get_group_id(0)];		\n";

				code << "	for (int n = 0; n < length; n++) {	\n" <<
						"		index = indices[indexStart + n];	\n" <<
						"		offK = offKVec[index];			\n" <<
						"		offX = offXVec[index];			\n" <<
						"		N = NVec[index];				\n";


				code << "	for (int stratum = 0; stratum < totalStrata; stratum++) {	\n" <<
						"		int stratumStart = NtoK[stratum];	\n" <<
						"		uint vecOffset = stratumStart*cvIndexStride + cvIndex;	\n" <<
						"		if (lid == 0) {					\n" <<
						"			weight = weightVector[vecOffset];	\n" <<
						"		}									\n" <<
						"		barrier(CLK_LOCAL_MEM_FENCE);		\n";
				code << "		if (weight != 0) {					\n";

				code << "		int total = NtoK[stratum+1] - stratumStart;	\n" <<
						"		int cases = casesVec[stratum];		\n" <<
						"		int controls = total - cases;		\n" <<
						"		int offKStrata = index*totalStrata + stratum;	\n";

	#ifdef USE_LOG_SUM
				code << "		B0[0][lid] = -INFINITY;					\n" <<
						"		B0[1][lid] = -INFINITY;					\n" <<
						"		B1[0][lid] = -INFINITY;					\n" <<
						"		B1[1][lid] = -INFINITY;					\n" <<
						"		B2[0][lid] = -INFINITY;					\n" <<
						"		B2[1][lid] = -INFINITY;					\n" <<
						"		if (lid == 0) {							\n" <<
						"			B0[0][lid] = 0;						\n" <<
						"			B0[1][lid] = 0;						\n" <<
						"		}										\n" <<
						"		const REAL logTwo = log((REAL)2.0);		\n";
	#else
				code << "		B0[0][lid] = 0;							\n" <<
						"		B0[1][lid] = 0;							\n" <<
						"		B1[0][lid] = 0;							\n" <<
						"		B1[1][lid] = 0;							\n" <<
						"		B2[0][lid] = 0;							\n" <<
						"		B2[1][lid] = 0;							\n" <<
						"		if (lid == 0) {							\n" <<
						"			B0[0][lid] = 1;						\n" <<
						"			B0[1][lid] = 1;						\n" <<
						"		}										\n";
	#endif
				//"	uint current = 0;						\n";

				code << "		uint loops;								\n" <<
						"		loops = cases / (TPB - 1);				\n" <<
						"		if (cases % (TPB - 1) > 0) {			\n" <<
						"			loops++;							\n" <<
						"		}										\n";

				// if loops == 1
				//code << "		if (loops == 1) {							\n" <<
				code << "			uint current = 0;						\n";

				if (formatType == INDICATOR || formatType == SPARSE) {
					code << "	__local uint currentKIndex, currentK;	\n" <<
							"	if (lid == 0) {							\n" <<
							"		currentKIndex = KStrata[offKStrata];					\n" <<
							"		if (currentKIndex == -1) {			\n" <<
							"			currentK = -1;					\n" <<
							"		} else {							\n" <<
							"			currentK = K[offK+currentKIndex];	\n" <<
							"		}									\n" <<
							"	}										\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);			\n";
				}

				if (formatType == INTERCEPT) {
					code << "REAL x;						\n";
				} else {
					code << "__local REAL x;				\n";
				}

				code << "	for (int col = 0; col < total; col++) {	\n" <<
#ifdef USE_LOG_SUM
					"		REAL U = xBetaVector[vecOffset + col * cvIndexStride];	\n";
#else
					"		REAL U = exp(xBetaVector[vecOffset + col * cvIndexStride]);	\n";
#endif

				if (formatType == DENSE) {
					code << "	if (lid == 0) {						\n" <<
							"		x = X[offX+stratumStart+col];			\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				if (formatType == INTERCEPT) {
	#ifdef USE_LOG_SUM
					code << "	x = 0;					\n";
	#else
					code << "	x = 1;					\n";
	#endif
				}

				if (formatType == INDICATOR) {
					code << "	if (lid == 0) {						\n" <<
							"		if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
	#ifdef USE_LOG_SUM
							"	x = -INFINITY;								\n" <<
	#else
							"	x = 0;								\n" <<
	#endif
							"		} else {	\n" <<
	#ifdef USE_LOG_SUM
							"	x = 0;								\n" <<
	#else
							"	x = 1;								\n" <<
	#endif
							"			currentKIndex++;			\n" <<
							"			currentK = K[offK + currentKIndex];	\n" <<
							"		}								\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				if (formatType == SPARSE) {
					code << "	if (lid == 0) {						\n" <<
							"		if (currentK == -1 || currentKIndex >= N || stratumStart + col != currentK) {			\n" <<
	#ifdef USE_LOG_SUM
							"	x = -INFINITY;								\n" <<
	#else
							"	x = 0;								\n" <<
	#endif
							"		} else {						\n" <<
							"			x = X[offX+currentKIndex];	\n" <<
							"			currentKIndex++;			\n" <<
							"			currentK = K[offK + currentKIndex];	\n" <<
							"		}								\n" <<
							"	}									\n" <<
							"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
				}

				code << "		if (lid > 0 && lid <= cases) {						\n" <<
	#ifdef USE_LOG_SUM
						//"			x = log(x);										\n" <<
						"			B0[current][lid] = log_sum(				 B0[1-current][lid], U+B0[1-current][lid-1]);	\n" <<
						"			B1[current][lid] = log_sum(log_sum(		 B1[1-current][lid], U+B1[1-current][lid-1]), x + U + B0[1-current][lid-1]);	\n" <<
						"			B2[current][lid] = log_sum(log_sum(log_sum(B2[1-current][lid], U+B2[1-current][lid-1]), x + U + B0[1-current][lid-1]), logTwo + x + U + B1[1-current][lid-1]);	\n" <<

	#else
						"			B0[current][lid] = B0[1-current][lid] + U*B0[1-current][lid-1];	\n" <<
						"			B1[current][lid] = B1[1-current][lid] + U*B1[1-current][lid-1] + x*U*B0[1-current][lid-1];	\n" <<
						"			B2[current][lid] = B2[1-current][lid] + U*B2[1-current][lid-1] + x*U*B0[1-current][lid-1] + 2*x*U*B1[1-current][lid-1];	\n" <<
	#endif
						"		}									\n" <<
						"		current = 1 - current;				\n" <<
						"		barrier(CLK_LOCAL_MEM_FENCE);		\n" <<
						"	}										\n";


				code << "	if (lid == cases) {							\n" <<
						"		REAL value0 = B0[1-current][lid];	\n" <<
						"		REAL value1 = B1[1-current][lid];	\n" <<
						"		REAL value2 = B2[1-current][lid];	\n" <<
	#ifdef USE_LOG_SUM
						"		grad += -exp(value1 - value0);			\n" <<
						"		hess += exp(2*(value1-value0)) - exp(value2 - value0);	\n" <<
	#else
						"		grad += -value1/value0;					\n" <<
						"		hess += value1*value1/value0/value0 - value2/value0;		\n" <<
	#endif
						"	}										\n";

				code << "	barrier(CLK_GLOBAL_MEM_FENCE);			\n";
				code << "	barrier(CLK_LOCAL_MEM_FENCE);			\n";
				code << "	}										\n";


				//code << "}											\n";


				code << "	if (lid == 0) {			\n";
				if (priorType == 1) {
					code << "lambda = priorParams[index];				\n";
				}
				if (priorType == 2) {
					code << "var = priorParams[index];				\n";
				}
				code << "	}										\n";
				code << "	barrier(CLK_LOCAL_MEM_FENCE);				\n";
				code << "	if (lid == 0) {	\n" <<
						"		uint offset = cvIndexStride*index+cvIndex;		\n" <<
						"		grad = grad - XjYVector[offset];	\n" <<
		    			"		REAL beta = betaVector[offset];		\n";

				if (priorType == 0) {
					code << " delta = -grad / hess;			\n";
				}
				if (priorType == 1) {
					//code << "	REAL lambda = priorParams[index];	\n" <<
					code << "	REAL negupdate = - (grad - lambda) / hess; \n" <<
							"	REAL posupdate = - (grad + lambda) / hess; \n" <<
							"	if (beta == 0 ) {					\n" <<
							"		if (negupdate < 0) {			\n" <<
							"			delta = negupdate;			\n" <<
							"		} else if (posupdate > 0) {		\n" <<
							"			delta = posupdate;			\n" <<
							"		} else {						\n" <<
							"			delta = 0;					\n" <<
							"		}								\n" <<
							"	} else {							\n" <<
							"		if (beta < 0) {					\n" <<
							"			delta = negupdate;			\n" <<
							"			if (beta+delta > 0) delta = -beta;	\n" <<
							"		} else {						\n" <<
							"			delta = posupdate;			\n" <<
							"			if (beta+delta < 0) delta = -beta;	\n" <<
							"		}								\n" <<
							"	}									\n";
				}
				if (priorType == 2) {
					//code << "	REAL var = priorParams[index];		\n" <<
					code << "	delta = - (grad + (beta / var)) / (hess + (1.0 / var));	\n";
				}

				code << "	REAL bound = boundVector[offset];		\n" <<
						"	if (delta < -bound)	{					\n" <<
						"		delta = -bound;						\n" <<
						"	} else if (delta > bound) {				\n" <<
						"		delta = bound;						\n" <<
						"	}										\n" <<
						"	REAL intermediate = max(fabs(delta)*2, bound/2);	\n" <<
						"	intermediate = max(intermediate, 0.001);	\n" <<
						"	boundVector[offset] = intermediate;		\n" <<
						//"	betaVector[offset] = delta + beta;		\n" <<
						"	if (delta != 0) {						\n" <<
						"		betaVector[offset] = delta + beta;	\n" <<
						"	}										\n";

				code << "	}										\n";

		        code << "   barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
		        		"	if (delta != 0)		{				\n" <<
		        		//"	if (scratchInt[0] > 0) {				\n" <<
		        		"	uint task = lid;						\n";

		        code <<	"	while (task < N) {		\n";
		        if (formatType == INDICATOR || formatType == SPARSE) {
		        	code << "  	uint k = K[offK + task];      	\n";
		        } else { // DENSE, INTERCEPT
		        	code << "   uint k = task;           		\n";
		        }
		        if (formatType == SPARSE || formatType == DENSE) {
		        	code << "   REAL inc = delta * X[offX + task]; \n";
		        } else { // INDICATOR, INTERCEPT
		        	code << "   REAL inc = delta;           	\n";
		        }
		        code << "		uint vecOffset = k*cvIndexStride + cvIndex;	\n" <<
		        		"		REAL xb;						\n" <<
						"		xb = xBetaVector[vecOffset] + inc;	\n" <<
		        		//"		REAL xb = xBetaVector[vecOffset] + inc;	\n" <<
						"		xBetaVector[vecOffset] = xb;	\n";
		        code << "		task += TPB;							\n";
		        code << "} 										\n";

		        code << "   barrier(CLK_GLOBAL_MEM_FENCE);           \n";
		        code << "}	\n";


		        code << "}	\n";
				code << "}	\n";
				code << "}	\n";
				return SourceCode(code.str(), name);
			}


	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForMMFindDeltaKernel(FormatType formatType, int priorType) {

		std::string name;
        if (priorType == 0) name = "doItAllMM" + getFormatTypeExtension(formatType) + "PriorNone";
        if (priorType == 1) name = "doItAllMM" + getFormatTypeExtension(formatType) + "PriorLaplace";
        if (priorType == 2) name = "doItAllMM" + getFormatTypeExtension(formatType) + "PriorNormal";

		std::stringstream code;
		code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

		code << "__kernel void " << name << "(            \n" <<
                "       __global const uint* offXVec,                  \n" <<
                "       __global const uint* offKVec,                  \n" <<
                "       __global const uint* NVec,                     \n" <<
				"       __global const REAL* X,           \n" <<
				"       __global const int* K,            \n" <<
				"       __global const REAL* Y,           \n" <<
				"		__global const REAL* Offs,		  \n" <<
				"       __global const REAL* xBetaVector,       \n" <<
				"       __global const REAL* expXBetaVector,    \n" <<
				"       __global const REAL* denomPidVector,	  \n" <<
				"       __global const int* pIdVector,           \n" <<  // TODO Make id optional
				"       __global const REAL* weightVector,	\n" <<
				"		__global const REAL* norm,			\n" <<
				"		__global REAL* boundVector,				\n" <<
				"		__global const REAL* priorParams,			\n" <<
				"		__global const REAL* XjYVector,			\n" <<
				"		__global REAL* betaVector,			\n" <<
				"		__global const int* doneVector,		\n" <<
				"		__global const int* indices,		\n" <<
				"		__global REAL* deltaVector,			\n" <<
				"		const uint cvIndexStride,		\n" <<
				"		const uint syncCVFolds,			\n" <<
				"		const uint J) {   		 	\n";    // TODO Make weight optional
		// Initialization

		code << "	uint lid0 = get_local_id(0);			\n" <<
				"	uint lid1 = get_local_id(1);			\n" <<
				"	uint task1 = lid1;						\n" <<
				"	uint index = indices[get_group_id(1)];	\n" <<
				"	uint cvIndex = get_group_id(0)*32+lid0;	\n" <<
				"	REAL sum0 = 0.0;						\n" <<
				"	REAL sum1 = 0.0;						\n" <<
				"	__local REAL grad[32][8];				\n" <<
				"	__local REAL hess[32][8];				\n" <<
				"	__local uint offK, offX, N;			\n" <<
				"	offK = offKVec[index];				\n" <<
				"	offX = offXVec[index];				\n" <<
				"	N = NVec[index];					\n" <<
				"	if (cvIndex < syncCVFolds) {		\n" <<
				"	while (task1 < N) {					\n";
		if (formatType == INDICATOR || formatType == SPARSE) {
			code << "  	uint k = K[offK + task1];      	\n";
		} else { // DENSE, INTERCEPT
			code << "   uint k = task1;           		\n";
		}
		if (formatType == SPARSE || formatType == DENSE) {
			code << "  	REAL x = X[offX + task1]; \n";
		} else { // INDICATOR, INTERCEPT
			// Do nothing
		}
		code << "		uint vecOffset = k*cvIndexStride;	\n" <<
				"		REAL exb = expXBetaVector[vecOffset+cvIndex];	\n" <<
				"		REAL numer = " << timesX("exb", formatType) << ";\n" <<
				"		REAL denom = denomPidVector[vecOffset+cvIndex];		\n" <<
				"		REAL w = weightVector[vecOffset+cvIndex];\n" <<
				"		REAL norm0 = norm[k];					\n";
		code << "       REAL gradient = w * numer / denom;       \n";
		code << "		REAL hessian = 0.0;			\n";
		if (formatType == INDICATOR || formatType == INTERCEPT) {
			code << "       hessian  = gradient  * norm0 / denom;				\n";
			//code << "       hessian  = " << weight("numer*norm0/denom/denom",useWeights) << ";\n";
		} else {
			code << "if (x != 0.0) { \n" <<
					//"		REAL nume2 = " << timesX("gradient", formatType) << "\n" <<
					//"		hessian = nume2 * norm0 / fabs(x) / denom " <<
					"       const REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
					"       hessian  = w* nume2 * norm0 / fabs(x) / denom / denom;  \n" <<
					"} \n";
		}
		code << "       sum0 += gradient; \n" <<
				"       sum1 += hessian;  \n";
		code << "       task1 += 8; \n" <<
				"   } \n" <<
				"	grad[lid0][lid1] = sum0;			\n" <<
				"	hess[lid0][lid1] = sum1;			\n" <<
				"	}									\n";

		code << "   for(int j = 1; j < 8; j <<= 1) {          \n" <<
	            "       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
	            "       uint mask = (j << 1) - 1;               \n" <<
	            "       if ((lid1 & mask) == 0) {                \n" <<
	            "           grad[lid0][lid1] += grad[lid0][lid1 + j]; \n" <<
	            "           hess[lid0][lid1] += hess[lid0][lid1 + j]; \n" <<
	            "       }                                       \n" <<
	            "   }                                           \n";

		code << "	if (lid1 == 0 && cvIndex < syncCVFolds) {	\n" <<
				"		uint offset = cvIndex*J+index;		\n" <<
				"		REAL grad0 = grad[lid0][lid1];		\n" <<
				"		grad0 = grad0 - XjYVector[offset];	\n" <<
				"		REAL hess0 = hess[lid0][lid1];		\n" <<
    			"		REAL beta = betaVector[offset];		\n";

		if (priorType == 0) {
			code << " REAL delta = -grad0 / hess0;			\n";
		}
		if (priorType == 1) {
			code << "	REAL lambda = priorParams[index];	\n" <<
					"	REAL delta;							\n" <<
					"	REAL negupdate = - (grad0 - lambda) / hess0; \n" <<
					"	REAL posupdate = - (grad0 + lambda) / hess0; \n" <<
					"	if (beta == 0 ) {					\n" <<
					"		if (negupdate < 0) {			\n" <<
					"			delta = negupdate;			\n" <<
					"		} else if (posupdate > 0) {		\n" <<
					"			delta = posupdate;			\n" <<
					"		} else {						\n" <<
					"			delta = 0;					\n" <<
					"		}								\n" <<
					"	} else {							\n" <<
					"		if (beta < 0) {					\n" <<
					"			delta = negupdate;			\n" <<
					"			if (beta+delta > 0) delta = -beta;	\n" <<
					"		} else {						\n" <<
					"			delta = posupdate;			\n" <<
					"			if (beta+delta < 0) delta = -beta;	\n" <<
					"		}								\n" <<
					"	}									\n";
		}
		if (priorType == 2) {
			code << "	REAL var = priorParams[index];		\n" <<
					"	REAL delta = - (grad0 + (beta / var)) / (hess0 + (1.0 / var));	\n";
		}

		code << "	delta = delta * doneVector[cvIndex];	\n" <<
				"	REAL bound = boundVector[offset];		\n" <<
				"	if (delta < -bound)	{					\n" <<
				"		delta = -bound;						\n" <<
				"	} else if (delta > bound) {				\n" <<
				"		delta = bound;						\n" <<
				"	}										\n" <<
				//"	REAL intermediate = 2;					\n" <<
				"	REAL intermediate = max(fabs(delta)*2, bound/2);	\n" <<
				"	intermediate = max(intermediate, 0.001);	\n" <<
				"	boundVector[offset] = intermediate;		\n" <<
				"	betaVector[offset] = delta + beta;		\n" <<
				"	deltaVector[index*cvIndexStride+cvIndex] = delta;					\n" <<
				"	}										\n";
		code << "}  \n"; // End of kernel
		return SourceCode(code.str(), name);
	}

	template <class BaseModel, typename WeightType, class BaseModelG>
	SourceCode
	    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForEmptyKernel() {
	        std::string name = "empty";
	        std::stringstream code;
	        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	        code << "__kernel void " << name << "( 	\n" <<
	        		"	const uint blah) {			\n" <<
					"	uint lid = get_local_id(0);	\n" <<
	        		"}   		 					\n";
	        return SourceCode(code.str(), name);
	    }




} // namespace bsccs

#endif /* KERNELS_HPP_ */
