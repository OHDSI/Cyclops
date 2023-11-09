/*
 * Kernels.hpp
 *
 *  Created on: Apr 4, 2016
 *      Author: msuchard
 */

#ifndef KERNELS_HPP_
#define KERNELS_HPP_

#include <boost/compute/type_traits/type_name.hpp>
#include "BaseKernels.hpp"
#include "ModelSpecifics.h"

namespace bsccs {

//namespace {
//
//template<typename T, bool isNvidiaDevice>
//struct ReduceBody1 {
//    static std::string body() {
//        std::stringstream k;
//        // local reduction for non-NVIDIA device
//        k <<
//            "   for(int j = 1; j < TPB; j <<= 1) {        \n" <<
//            "       barrier(CLK_LOCAL_MEM_FENCE);         \n" <<
//            "       uint mask = (j << 1) - 1;             \n" <<
//            "       if ((lid & mask) == 0) {              \n" <<
//            "           scratch[lid] += scratch[lid + j]; \n" <<
//            "       }                                     \n" <<
//            "   }                                         \n";
//        return k.str();
//    }
//};
//
//template<typename T, bool isNvidiaDevice>
//struct ReduceBody2 {
//    static std::string body() {
//        std::stringstream k;
//        // local reduction for non-NVIDIA device
//        k <<
//            "   for(int j = 1; j < TPB; j <<= 1) {          \n" <<
//            "       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
//            "       uint mask = (j << 1) - 1;               \n" <<
//            "       if ((lid & mask) == 0) {                \n" <<
//            "           scratch[0][lid] += scratch[0][lid + j]; \n" <<
//            "           scratch[1][lid] += scratch[1][lid + j]; \n" <<
//            "       }                                       \n" <<
//            "   }                                           \n";
//        return k.str();
//    }
//};
//
//template<typename T, bool isNvidiaDevice>
//struct ReduceBody4 {
//    static std::string body() {
//        std::stringstream k;
//        // local reduction for non-NVIDIA device
//        k <<
//            "   for(int j = 1; j < TPB; j <<= 1) {          \n" <<
//            "       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
//            "       uint mask = (j << 1) - 1;               \n" <<
//            "       if ((lid & mask) == 0) {                \n" <<
//            "           scratch[0][lid] += scratch[0][lid + j]; \n" <<
//            "           scratch[1][lid] += scratch[1][lid + j]; \n" <<
//			"			scratch[2][lid] += scratch[2][lid + j];	\n" <<
//			"			scratch[3][lid] += scratch[3][lid + j];	\n" <<
//            "       }                                       \n" <<
//            "   }                                           \n";
//        return k.str();
//    }
//};
//
//template<typename T>
//struct ReduceBody1<T,true>
//{
//    static std::string body() {
//        std::stringstream k;
//        k <<
//            "   barrier(CLK_LOCAL_MEM_FENCE);\n" <<
//            "   if (TPB >= 1024) { if (lid < 512) { sum += scratch[lid + 512]; scratch[lid] = sum; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
//            "   if (TPB >=  512) { if (lid < 256) { sum += scratch[lid + 256]; scratch[lid] = sum; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
//            "   if (TPB >=  256) { if (lid < 128) { sum += scratch[lid + 128]; scratch[lid] = sum; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
//            "   if (TPB >=  128) { if (lid <  64) { sum += scratch[lid +  64]; scratch[lid] = sum; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
//        // warp reduction
//            "   if (lid < 32) { \n" <<
//        // volatile this way we don't need any barrier
//            "       volatile __local TMP_REAL *lmem = scratch;                 \n" <<
//            "       if (TPB >= 64) { lmem[lid] = sum = sum + lmem[lid + 32]; } \n" <<
//            "       if (TPB >= 32) { lmem[lid] = sum = sum + lmem[lid + 16]; } \n" <<
//            "       if (TPB >= 16) { lmem[lid] = sum = sum + lmem[lid +  8]; } \n" <<
//            "       if (TPB >=  8) { lmem[lid] = sum = sum + lmem[lid +  4]; } \n" <<
//            "       if (TPB >=  4) { lmem[lid] = sum = sum + lmem[lid +  2]; } \n" <<
//            "       if (TPB >=  2) { lmem[lid] = sum = sum + lmem[lid +  1]; } \n" <<
//            "   }                                                            \n";
//        return k.str();
//    }
//};
//
//template<typename T>
//struct ReduceBody2<T,true>
//{
//    static std::string body() {
//        std::stringstream k;
//        k <<
//            "   barrier(CLK_LOCAL_MEM_FENCE); \n" <<
//            "   if (TPB >= 1024) { if (lid < 512) { sum0 += scratch[0][lid + 512]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 512]; scratch[1][lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
//            "   if (TPB >=  512) { if (lid < 256) { sum0 += scratch[0][lid + 256]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 256]; scratch[1][lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
//            "   if (TPB >=  256) { if (lid < 128) { sum0 += scratch[0][lid + 128]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 128]; scratch[1][lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
//            "   if (TPB >=  128) { if (lid <  64) { sum0 += scratch[0][lid +  64]; scratch[0][lid] = sum0; sum1 += scratch[1][lid +  64]; scratch[1][lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
//        // warp reduction
//            "   if (lid < 32) { \n" <<
//        // volatile this way we don't need any barrier
//            "       volatile __local TMP_REAL **lmem = scratch; \n" <<
//            "       if (TPB >= 64) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+32]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+32]; } \n" <<
//            "       if (TPB >= 32) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+16]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+16]; } \n" <<
//            "       if (TPB >= 16) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 8]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 8]; } \n" <<
//            "       if (TPB >=  8) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 4]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 4]; } \n" <<
//            "       if (TPB >=  4) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 2]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 2]; } \n" <<
//            "       if (TPB >=  2) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 1]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 1]; } \n" <<
//            "   }                                            \n";
//        return k.str();
//    }
//};
//
//template<typename T>
//struct ReduceBody4<T,true>
//{
//    static std::string body() {
//        std::stringstream k;
//        k <<
//            "   barrier(CLK_LOCAL_MEM_FENCE); \n" <<
//            "   if (TPB >= 1024) { if (lid < 512) { sum0 += scratch[0][lid + 512]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 512]; scratch[1][lid] = sum1; \n" <<
//			"										sum2 += scratch[2][lid + 512]; scratch[2][lid] = sum2; sum3 += scratch[3][lid + 512]; scratch[3][lid] = sum3; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
//            "   if (TPB >=  512) { if (lid < 256) { sum0 += scratch[0][lid + 256]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 256]; scratch[1][lid] = sum1; \n" <<
//			"										sum2 += scratch[2][lid + 256]; scratch[2][lid] = sum2; sum3 += scratch[3][lid + 256]; scratch[3][lid] = sum3; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
//            "   if (TPB >=  256) { if (lid < 128) { sum0 += scratch[0][lid + 128]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 128]; scratch[1][lid] = sum1; \n" <<
//			"										sum2 += scratch[2][lid + 128]; scratch[2][lid] = sum2; sum3 += scratch[3][lid + 128]; scratch[3][lid] = sum3; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
//            "   if (TPB >=  128) { if (lid <  64) { sum0 += scratch[0][lid +  64]; scratch[0][lid] = sum0; sum1 += scratch[1][lid +  64]; scratch[1][lid] = sum1; \n" <<
//			"										sum2 += scratch[2][lid +  64]; scratch[2][lid] = sum2; sum3 += scratch[3][lid +  64]; scratch[3][lid] = sum3; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
//        // warp reduction
//            "   if (lid < 32) { \n" <<
//        // volatile this way we don't need any barrier
//            "       volatile __local TMP_REAL **lmem = scratch; \n" <<
//            "       if (TPB >= 64) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+32]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+32]; lmem[2][lid] = sum2 = sum2 + lmem[2][lid+32]; lmem[3][lid] = sum3 = sum3 + lmem[3][lid+32];} \n" <<
//            "       if (TPB >= 32) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+16]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+16]; lmem[2][lid] = sum2 = sum2 + lmem[2][lid+16]; lmem[3][lid] = sum3 = sum3 + lmem[3][lid+16];} \n" <<
//            "       if (TPB >= 16) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 8]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 8]; lmem[2][lid] = sum2 = sum2 + lmem[2][lid+ 8]; lmem[3][lid] = sum3 = sum3 + lmem[3][lid+ 8];} \n" <<
//            "       if (TPB >=  8) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 4]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 4]; lmem[2][lid] = sum2 = sum2 + lmem[2][lid+ 4]; lmem[3][lid] = sum3 = sum3 + lmem[3][lid+ 4];} \n" <<
//            "       if (TPB >=  4) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 2]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 2]; lmem[2][lid] = sum2 = sum2 + lmem[2][lid+ 2]; lmem[3][lid] = sum3 = sum3 + lmem[3][lid+ 2];} \n" <<
//            "       if (TPB >=  2) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 1]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 1]; lmem[2][lid] = sum2 = sum2 + lmem[2][lid+ 1]; lmem[3][lid] = sum3 = sum3 + lmem[3][lid+ 1];} \n" <<
//            "   }                                            \n";
//        return k.str();
//    }
//};
//
//template <class BaseModel> // if BaseModel derives from IndependentData
//typename std::enable_if<std::is_base_of<IndependentData,BaseModel>::value, std::string>::type
//static group(const std::string& id, const std::string& k) {
//    return k;
//};
//
//template <class BaseModel> // if BaseModel does not derive from IndependentData
//typename std::enable_if<!std::is_base_of<IndependentData,BaseModel>::value, std::string>::type
//static group(const std::string& id, const std::string& k) {
//    return id + "[" + k + "]";
//};
//}; // anonymous namespace

template <class BaseModel, typename RealType, class BaseModelG>
SourceCode
GpuModelSpecifics<BaseModel, RealType, BaseModelG>::writeCodeForGradientHessianKernel(FormatType formatType, bool useWeights, bool isNvidia) {

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

	code << "		REAL xb = xBeta[k];			\n" <<
			// needs offs later for SCCS
			"		REAL exb = " << BaseModelG::getOffsExpXBetaG("0","xb") << " ;\n" <<
			//"       REAL exb = expXBeta[k];     \n" <<
			"       REAL numer = " << timesX("exb", formatType) << ";\n";
	//"       const REAL denom = 1.0 + exb;			\n";
	if (BaseModelG::logisticDenominator) {
		code << "REAL denom = 1.0 + exb;			\n";
	} else {
		code << "REAL denom = denominator[" << BaseModelG::getGroupG("id", "k") << "];\n";
	}
	//code << "		const REAL denom = denominator[k];		\n";
	if (useWeights) {
		code << "       REAL w = weight[k];\n";
	}

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
	code << (isNvidia ? ReduceBody2<RealType,true>::body() : ReduceBody2<RealType,false>::body());
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

// CV nonstratified CCD
	template <class BaseModel, typename RealType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, RealType, BaseModelG>::writeCodeForSyncCVGradientHessianKernel(FormatType formatType, bool isNvidia, bool layoutByPerson) {

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
				"       __global REAL* buffer,            \n" <<
				"       __global const int* pIdVector,           \n" <<  // TODO Make id optional
				"       __global const REAL* weightVector,	\n" <<
				"		const uint cvIndexStride,		\n" <<
				"		const uint KStride,				\n" <<
				"		const uint syncCVFolds,			\n" <<
				"		__global int* allZero) {   		 	\n";    // TODO Make weight optional
		// Initialization
		code << "	if (get_global_id(0) == 0) allZero[0] = 1;	\n" <<
				"	uint lid0 = get_local_id(0);		\n" <<
				"	uint lid1 = get_local_id(1);		\n" <<
				"	uint cvIndex = get_global_id(0);	\n" <<
		        "   uint gid1 = get_global_id(1);             \n" <<
				"	uint loopSize = get_global_size(1);	\n" <<
				"	uint task1 = gid1;							\n" <<
				"	uint mylid = lid1*TPB0+lid0;			\n" <<
				"	__local REAL scratch[2][TPB];		\n" <<
				"	REAL sum0 = 0.0;					\n" <<
				"	REAL sum1 = 0.0;					\n";

		code << "	if (cvIndex < syncCVFolds) {		\n";
		code << "	while (task1 < N) {					\n";

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

		if (layoutByPerson) {
			code << "	uint vecOffset = k * cvIndexStride + cvIndex;	\n";
		} else {
			code << "	uint vecOffset = k + KStride * cvIndex;	\n";
		}
		code << "		REAL xb = xBetaVector[vecOffset];			\n" <<
        		"		REAL exb = " << BaseModelG::getOffsExpXBetaG("0", "xb") << ";\n" <<
				"		REAL numer = " << timesX("exb", formatType) << ";\n";
		code << " 		REAL denom = (REAL)1.0 + exb;				\n";

		code << "		REAL w = weightVector[vecOffset];\n";
		code << BaseModelG::incrementGradientAndHessianG(formatType, true);
		code << "       sum0 += gradient; \n" <<
				"       sum1 += hessian;  \n";
		code << "       task1 += loopSize; \n" <<
				"   } \n";


		code << "	scratch[0][mylid] = sum0;	\n" <<
				"	scratch[1][mylid] = sum1;	\n";

		if (layoutByPerson) {
		code << "   for(int j = 1; j < TPB1; j <<= 1) {          \n" <<
	            "       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
	            "       uint mask = (j << 1) - 1;               \n" <<
	            "       if ((lid1 & mask) == 0) {                \n" <<
	            "           scratch[0][mylid] += scratch[0][mylid + TPB0 * j]; \n" <<
	            "           scratch[1][mylid] += scratch[1][mylid + TPB0 * j]; \n" <<
	            "       }                                       \n" <<
	            "   }                                         	\n";
		} else {
			code << "	uint lid = lid1;					\n";
			code << (isNvidia ? ReduceBody2<RealType,true>::body() : ReduceBody2<RealType,false>::body());
		}

	code << "	if (lid1 == 0) {					\n" <<
			"		buffer[get_global_id(0) * get_num_groups(1) + get_group_id(1)] = scratch[0][mylid];	\n" <<
			"		buffer[(get_global_id(0) + syncCVFolds) * get_num_groups(1) + get_group_id(1)] = scratch[1][mylid];	\n" <<
			"	}									\n";

		code << "	}										\n" <<
				"	}									\n";
		//code << "}  \n"; // End of kernel
		return SourceCode(code.str(), name);
	}

template <class BaseModel, typename RealType, class BaseModelG>
SourceCode
GpuModelSpecifics<BaseModel, RealType, BaseModelG>::writeCodeForProcessDeltaKernel(int priorType) {
	std::string name;
	if (priorType == 0) name = "ProcessDeltaKernelNone";
	if (priorType == 1) name = "ProcessDeltaKernelLaplace";
	if (priorType == 2) name = "ProcessDeltaKernelNormal";

	std::stringstream code;
	code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	code << "__kernel void " << name << "(            \n" <<
			"		__global const REAL* buffer,				\n" <<
			"		__global REAL* deltaVector,			\n" <<
			"		const uint wgs,						\n" <<
			"		__global REAL* boundVector,				\n" <<
			"		__global const REAL* priorParams,			\n" <<
			"		__global const REAL* XjYVector,			\n" <<
			"		const uint index,					\n" <<
			"		__global REAL* betaVector,			\n" <<
			"		const uint totalStrata) {    \n";    // TODO Make weight optional
	// Initialization
	code <<	"	__local REAL scratch[2][TPB];				\n" <<
			"	uint lid = get_local_id(0);				\n" <<
			"	scratch[0][lid] = 0.0;					\n" <<
			"	scratch[1][lid] = 0.0;					\n" <<
			//"	uint k = totalStrata;					\n" <<
			"	uint k = TPB;							\n" <<
			"	uint task = lid;						\n" <<
			"	while (task < wgs) {					\n" <<
			"		scratch[0][lid] += buffer[lid];		\n" <<
			"		scratch[1][lid] += buffer[lid+wgs];	\n" <<
			"		task += TPB;						\n" <<
			//"		printf(\"(s %d g %f h %f)\",lid, scratch[0][lid], scratch[1][lid]);\n" <<
			"	}										\n" <<
			//"	if (k > wgs) k = wgs;					\n" <<
			//"	if (lid < k) {						\n" <<
			//"		scratch[0][lid] = buffer[lid];	\n" <<
			//"		scratch[1][lid] = buffer[lid+wgs];	\n" <<
			//"		printf(\"(s %d g %f h %f)\",lid, scratch[0][lid], scratch[1][lid]);\n" <<
			//"	}										\n" <<
			"   for(int j = 1; j < k; j <<= 1) {          \n" <<
			"       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
			"       uint mask = (j << 1) - 1;               \n" <<
			"       if ((lid & mask) == 0) {                \n" <<
			"           scratch[0][lid] += scratch[0][lid + j]; \n" <<
			"           scratch[1][lid] += scratch[1][lid + j]; \n" <<
			"       }                                       \n" <<
			"   }                                           \n";

	code << "	if (lid == 0) {							\n" <<
			"		__local uint offset;				\n" <<
			"		offset = index;			\n" <<
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

	code << "	REAL bound = boundVector[offset];		\n" <<
			"	if (delta < -bound)	{					\n" <<
			"		delta = -bound;						\n" <<
			"	} else if (delta > bound) {				\n" <<
			"		delta = bound;						\n" <<
			"	}										\n" <<
			//"	REAL intermediate = 2;					\n" <<
			"	REAL intermediate = max(fabs(delta)*2, bound/2);	\n" <<
			"	intermediate = max(intermediate, 0.001);	\n" <<
			"	boundVector[offset] = intermediate;		\n" <<
			"	deltaVector[index] = delta;	\n" <<
			"	betaVector[offset] = delta + beta;		\n" <<
			//"	printf(\"delta %d: %f\", index, delta);		\n" <<

			"	}										\n";

	code << "	}										\n";
	return SourceCode(code.str(), name);
}

// step between compute grad hess and update XB
template <class BaseModel, typename RealType, class BaseModelG>
SourceCode
GpuModelSpecifics<BaseModel, RealType, BaseModelG>::writeCodeForProcessDeltaSyncCVKernel(int priorType) {
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
			"	scratch[0][lid] = 0.0;					\n" <<
			"	scratch[1][lid] = 0.0;					\n" <<
			"	if (lid < wgs) {						\n" <<
			"		scratch[0][lid] = buffer[lid + wgs * cvIndex];	\n" <<
			"		scratch[1][lid] = buffer[lid + wgs * (cvIndex + syncCVFolds)];	\n" <<
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
    		"		__local uint offset;				\n";
//    if (layoutByPerson) {
//		code << "		offset = index * cvIndexStride + cvIndex;			\n";
//    } else {
		code << "		offset = index + J * cvIndex;			\n";
//    }
					code << "		__local REAL grad, hess, beta, delta;		\n" <<
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


// update nonstratified
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
                "       __global const REAL* deltaVector,          \n" <<
                "       __global const REAL* X,    \n" <<
                "       __global const int* K,     \n" <<
                "       __global const REAL* Y,    \n" <<
                "       __global REAL* xBeta,      \n" <<
                "       __global REAL* expXBeta,   \n" <<
                "       __global REAL* denominator,\n" <<
                "       __global const int* id,		\n" <<
				"		const uint index) {   \n";
        code << "   uint task = get_global_id(0); \n" <<
        		"	REAL delta = deltaVector[index];	\n";
        code << "	const uint loopSize = get_global_size(0); \n";

        code << "	while (task < N) {				\n";

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

        code << "       REAL xb = xBeta[k] + inc; 		\n" <<
                "       xBeta[k] = xb;                  \n";
        code << "		task += loopSize;				\n";

        if (BaseModel::likelihoodHasDenominator) {
            // TODO: The following is not YET thread-safe for multi-row observations
            // code << "       const REAL oldEntry = expXBeta[k];                 \n" <<
            //         "       const REAL newEntry = expXBeta[k] = exp(xBeta[k]); \n" <<
            //         "       denominator[" << group<BaseModel>("id","k") << "] += (newEntry - oldEntry); \n";
            // code << "       REAL exb = " << BaseModelG::getOffsExpXBetaG() << " ;\n" <<
                    //"       expXBeta[k] = exb;        \n" <<
					//"		denominator[k] = 1.0 + exb; \n";
        	//code << "expXBeta[k] = exp(xb); \n";
        	//code << "expXBeta[k] = exp(1); \n";
        }

        code << "   } \n";
        code << "}    \n";

        return SourceCode(code.str(), name);
    }

	// CV update XB
	template <class BaseModel, typename RealType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, RealType, BaseModelG>::writeCodeForSyncUpdateXBetaKernel(FormatType formatType, bool layoutByPerson) {

        std::string name = "updateXBetaSync" + getFormatTypeExtension(formatType);

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(     \n" <<
                "       const uint offX,           \n" <<
                "       const uint offK,           \n" <<
				"		const uint N,				\n" <<
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
				"		const uint KStride,			\n" <<
				"		const uint syncCVFolds,		\n" <<
				"		const uint index,			\n" <<
				"		__global const int* allZero) {   \n";

        code << "	uint lid0 = get_local_id(0);		\n" <<
        		"	uint lid1 = get_local_id(1);		\n" <<
        		"	if (allZero[0] == 0) {				\n";
        code << "	uint mylid = lid1 * TPB0 + lid0;	\n" <<
        		"	uint task1 = get_global_id(1);			\n" <<
				"	uint cvIndex = get_global_id(0);	\n" <<
				"	uint loopSize = get_global_size(1);	\n" <<
				"	if (cvIndex < syncCVFolds) {		\n";
        code << "		REAL delta = deltaVector[index * cvIndexStride + cvIndex];	\n";// <<
        code << "	if (delta != 0) {					\n";

        code << "	while (task1 < N) {					\n";
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
        if (layoutByPerson) {
        	code << "	uint vecOffset = k*cvIndexStride + cvIndex;	\n";
        } else {
        	code << "	uint vecOffset = k + KStride * cvIndex;		\n";
        }
        code << "		REAL xb = xBetaVector[vecOffset] + inc;	\n" <<
				"		xBetaVector[vecOffset] = xb;	\n";
        code << "		task1 += loopSize;				\n";
        code << "	}									\n";


        code << "   } \n";
        code << "}	\n";
        code << "}    \n";
        code << "}		\n";

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
        	code << "const REAL xb = xBeta[task];	\n" <<
            		"REAL exb = " << BaseModelG::getOffsExpXBetaG("0","xb") << " ;\n" <<
					"expXBeta[task] = exb;			\n";
        } else if (BaseModel::likelihoodHasDenominator) {
        	code << "const REAL xb = xBeta[task];\n" <<
        			//"const REAL y = Y[task];\n" <<
					"const REAL offs = Offs[task];\n";
					//"const int k = task;";
        	code << "REAL exb = " << BaseModelG::getOffsExpXBetaG("0","xb") << ";\n";
        	code << "expXBeta[task] = exb;\n";
    		code << "denominator[" << BaseModelG::getGroupG("id", "task") << "] =" << BaseModelG::getDenomNullValueG() << "+ exb;\n";
        	//code << "denominator[task] = (REAL)1.0 + exb;\n";
        }
        code << "   } \n";
        code << "}    \n";
        return SourceCode(code.str(), name);
    }

// CV CRS
	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncComputeRemainingStatisticsKernel(bool layoutByPerson) {

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
				"		const uint KStride,			\n" <<
				"		const uint syncCVFolds,		\n" <<
				"		const uint N) {   \n";

        code << "		uint lid0 = get_local_id(0);	\n" <<
        		"		uint lid1 = get_local_id(1);	\n" <<
				"		uint cvIndex = get_global_id(0);	\n" <<
				"		uint gid1 = get_global_id(1);	\n" <<
				"		uint loopSize = get_global_size(1);	\n" <<
				"		uint task1 = gid1;				\n";

        code << "		if (cvIndex < syncCVFolds) {	\n" <<
        		"			while (task1 < N) {			\n";

        if (layoutByPerson) {
        	code << "		uint vecOffset = task1 * cvIndexStride + cvIndex;	\n";
        } else {
        	code << "		uint vecOffset = task1 + KStride * cvIndex;			\n";
        }
        code << "			REAL xb = xBetaVector[vecOffset];\n";
        code << "			REAL exb = " << BaseModelG::getOffsExpXBetaG("0", "xb") << ";\n";
        code << "			expXBetaVector[vecOffset] = exb;\n";

        //"const REAL y = Y[task];\n" <<
        //"const REAL offs = Offs[task];\n";

        if (BaseModel::likelihoodHasDenominator) {
        	if (layoutByPerson) {
        		code << "	denomPidVector[" << BaseModelG::getGroupG("pIdVector", "task1") << " * cvIndexStride + cvIndex] =" << BaseModelG::getDenomNullValueG() << "+ exb;\n";
        	} else {
        		code << "	denomPidVector[" << BaseModelG::getGroupG("pIdVector", "task1") << " + KStride * cvIndex] =" << BaseModelG::getDenomNullValueG() << "+ exb;\n";
        	}
        }
        	code << "		task1 += loopSize;	\n";

        code << "   		} \n";
        code << "		}    \n";

        code << "	}    \n";

        return SourceCode(code.str(), name);
    }


// stratified CRS
template <class BaseModel, typename RealType, class BaseModelG>
SourceCode
GpuModelSpecifics<BaseModel, RealType, BaseModelG>::writeCodeForStratifiedComputeRemainingStatisticsKernel(bool efron) {

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
            "       __global const int* id,		\n" <<
			"		__global const int* NtoK,	\n" <<
			"		__global const REAL* NWeight,	\n" <<
			"		__global REAL* denominator2) {   \n";
    code << "   uint lid = get_local_id(0); 		\n" <<
    		"	uint stratum = get_group_id(0);		\n" <<
			"	__local uint start, end;			\n" <<
			"	__local REAL stratumWeight;			\n" <<
			"	if (lid == 0) {						\n" <<
			"		start = NtoK[stratum];			\n" <<
			"		end = NtoK[stratum+1];			\n" <<
			"		stratumWeight = NWeight[stratum];	\n" <<
			"	}									\n" <<
			"	barrier(CLK_LOCAL_MEM_FENCE);		\n";
    if (efron) {
    	code << "	__local REAL scratch[2][TPB];	\n" <<
    			"	REAL sum = 0.0;					\n" <<
				"	REAL sum1 = 0.0;				\n";
    } else {
    	code << "	__local REAL scratch[TPB];			\n" <<
    			"	REAL sum = 0.0;						\n";
    }

    code << "	uint task = start + lid;			\n";
    //        "   const uint loopSize = get_global_size(0); \n";
    // Local and thread storage
    code << "   while (task < end) {      			\n";
    code << "		const REAL xb = xBeta[task];	\n" <<
    			//"const REAL y = Y[task];\n" <<
				//"const int k = task;";
    		"		REAL exb = " << BaseModelG::getOffsExpXBetaG("0","xb") << ";\n" <<
			"		expXBeta[task] = exb;			\n" <<
			"		sum += exb;						\n";
    if (efron) {
    	code << "	sum1 += exb * Y[task];			\n";
    }
    code << "		task += TPB;					\n" <<
			"	}									\n";
    if (efron) {
    	code << "	scratch[0][lid] = sum;			\n" <<
    			"	scratch[1][lid] = sum1;			\n";
        code << ReduceBody2<RealType,false>::body();
        code << "	if (lid == 0) {						\n" <<
        		"		denominator[stratum] = scratch[0][0];\n" <<// * stratumWeight;	\n" <<
				"		denominator2[stratum] = scratch[1][0];	\n" <<
				"	}									\n";
    } else {
    	code << "	scratch[lid] = sum;				\n";
        code << ReduceBody1<RealType,false>::body();
        code << "	if (lid == 0) {						\n" <<
        		"		denominator[stratum] = scratch[0];\n" <<// * stratumWeight;	\n" <<
				"	}									\n";
    }

    code << "}    \n";
    return SourceCode(code.str(), name);
}

template <class BaseModel, typename RealType, class BaseModelG>
SourceCode
GpuModelSpecifics<BaseModel, RealType, BaseModelG>::writeCodeForComputeXjYKernel(FormatType formatType, bool layoutByPerson) {
	std::string name = "computeXjY";

	std::stringstream code;

	code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	code << "__kernel void " << name << "(            	\n" <<
			"	__global const uint* offXVec,                  \n" <<
			"   __global const uint* offKVec,                  \n" <<
			"   __global const uint* NVec,                     \n" <<
			//"       const uint offX,                  \n" <<
			//"       const uint offK,                  \n" <<
			//"       const uint N,                     \n" <<
			"   __global const REAL* X,           \n" <<
			"   __global const int* K,            \n" <<
			"	__global const REAL* Y,					\n" <<
			"	__global const REAL* weightVector,		\n" <<
			"	__global REAL* XjYVector,	\n" <<
			"	__const uint cvIndexStride,				\n" <<
			"	__const uint KStride,					\n" <<
			"	const uint J,							\n" <<
			"	const uint length,						\n" <<
			"	__global const int* indices)	{	\n" <<
			"	__local REAL scratch[TPB];				\n" <<
			"	uint cvIndex = get_group_id(0);			\n" <<
			"	uint lid = get_local_id(0);				\n";

	code << "	for (int ind = 0; ind < length; ind++) {	\n" <<
			"		int index = indices[ind];			\n" <<
			"		uint offK = offKVec[index];			\n" <<
			"		uint offX = offXVec[index];			\n" <<
			"		uint N = NVec[index];				\n" <<
			"		uint task = lid;				\n" <<
			"		REAL sum = 0.0;							\n";

	code << "		while (task < N) {						\n";
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

	if (layoutByPerson) {
		code << "		uint vecOffset = k * cvIndexStride + cvIndex;	\n";
	} else {
		code << "		uint vecOffset = k + KStride * cvIndex;	\n";
	}

	code << "			sum += " << timesX("Y[k] * weightVector[vecOffset]", formatType) << ";\n";

	code << "			task += TPB;						\n";
	code << "		}										\n";

	code << "		scratch[lid] = sum;						\n";

	code << ReduceBody1<RealType,false>::body();

	code << "		if (lid == 0) {							\n";
//	if (layoutByPerson) {
//		code << "		uint vecOffset = index * cvIndexStride + cvIndex; \n";
//	} else {
		code << "		uint vecOffset = cvIndex * J + index;			\n";
//	}
	code << "			XjYVector[vecOffset] = scratch[lid];		\n";
	code << "		}										\n";

	code << "		barrier(CLK_GLOBAL_MEM_FENCE);			\n";

	code << "	}										\n";


	code << "	}										\n";


	return SourceCode(code.str(), name);
}

// not being used. working?
template <class BaseModel, typename RealType, class BaseModelG>
SourceCode
GpuModelSpecifics<BaseModel, RealType, BaseModelG>::writeCodeForGetLogLikelihood(bool useWeights, bool isNvidia) {
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
    code << " sum += " << weightK(BaseModelG::logLikeNumeratorContribG("y","xb"), useWeights) << ";\n";

    if (BaseModel::likelihoodHasDenominator) {
    	code << "if (task < N) {	\n";
    	code << " const REAL wN = Nweight[task];\n";
    	if (BaseModel::cumulativeGradientAndHessian) {
    		code << "const REAL denom = accDenominator[task];\n";
    	} else {
    		code << "const REAL denom = denominator[task];\n";
    	}
    	code << "sum -= (REAL) " << BaseModelG::logLikeDenominatorContribG("wN", "denom") << ";\n";
    	code << "}\n";
    }

    // Bookkeeping
    code << "       task += loopSize; \n" <<
    		"   } \n";
			// Thread -> local
			code << "   scratch[lid] = sum; \n";
    code << (isNvidia ? ReduceBody1<RealType,true>::body() : ReduceBody1<RealType,false>::body());

    code << "   if (lid == 0) { \n" <<
    		"       buffer[get_group_id(0)] = scratch[0]; \n" <<
			"   } \n";

    code << "}  \n"; // End of kernel
    return SourceCode(code.str(), name);
}


template <class BaseModel, typename RealType, class BaseModelG>
SourceCode
GpuModelSpecifics<BaseModel, RealType, BaseModelG>::writeCodeForGetPredLogLikelihood(bool layoutByPerson) {
	std::string name = "predLogLikelihood";

	std::stringstream code;

	code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	code << "__kernel void " << name << "(            	\n" <<
			"	const uint N,							\n" <<
			"	const uint cvIndex,						\n" <<
			"	__global const REAL* XBetaVector,		\n" <<
			"	__global const REAL* Y,					\n" <<
			"	__global const REAL* weights,			\n" <<
			"	__global const REAL* denomPidVector,	\n" <<
			"	__const uint cvIndexStride,				\n" <<
			"	__const uint KStride,					\n" <<
			"	__global REAL* buffer)	{	\n" <<
			"	__local REAL scratch[TPB];				\n" <<
			"	uint lid = get_local_id(0);				\n" <<
			"	uint task = lid;						\n" <<
			"	REAL sum = 0.0;							\n" <<
			"	while (task < N) {						\n";
	if (layoutByPerson) {
		code << "	uint vecOffset = task * cvIndexStride + cvIndex;	\n";
		code << "	sum += (XBetaVector[vecOffset] ) * Y[task] * weights[task]- log(denomPidVector[" << BaseModelG::getGroupG("pIdVector", "task") << " * cvIndexStride + cvIndex])* weights[task]; \n";

	} else {
		code << "	uint vecOffset = task + KStride * cvIndex;	\n";
		code << "	sum += (XBetaVector[vecOffset] ) * Y[task] * weights[task]- log(denomPidVector[" << BaseModelG::getGroupG("pIdVector", "task") << " + KStride * cvIndex])* weights[task]; \n";
	}
	code << "		task += TPB;						\n";
	code << "	}										\n";
	code << "	scratch[lid] = sum;						\n";

	code << ReduceBody1<RealType,false>::body();

	code << "	if (lid == 0) {							\n" <<
			"		buffer[0] = scratch[lid];		\n" <<
			"	}										\n";

	code << "	}										\n";

	return SourceCode(code.str(), name);
}


template <class BaseModel, typename RealType, class BaseModelG>
SourceCode
GpuModelSpecifics<BaseModel, RealType, BaseModelG>::writeCodeForDoItAllNoSyncCVKernel(FormatType formatType, int priorType) {

	std::string name;
	if (priorType == 0) name = "doItAllNoSyncCV" + getFormatTypeExtension(formatType) + "PriorNone";
	if (priorType == 1) name = "doItAllNoSyncCV" + getFormatTypeExtension(formatType) + "PriorLaplace";
	if (priorType == 2) name = "doItAllNoSyncCV" + getFormatTypeExtension(formatType) + "PriorNormal";

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
			//"		const uint syncCVFolds,			\n" <<
			//"		const uint index)	{				\n";
			"		const uint indexStart,				\n" <<
			"		const uint length,				\n" <<
			"		__global const uint* indices) {   		 	\n";    // TODO Make weight optional
	// Initialization
	code << "	__local uint offK, offX, N, index, cvIndex, cvIndexStride;	\n" <<
			//"	if (get_global_id(0)==0) printf(\"tpb = %d \", TPB);	\n" <<
			"	__local REAL scratch[2][TPB];			\n" <<
			"	__local REAL delta;					\n" <<
			//"	__local REAL localXB[TPB*3];	\n" <<
			"	uint lid = get_local_id(0);			\n" <<
			"	cvIndex = 0;						\n" <<
			"	cvIndexStride = 1;					\n";

	code << "	for (int n = 0; n < length; n++) {	\n" <<
			"		index = indices[indexStart + n];	\n" <<
			"		offK = offKVec[index];			\n" <<
			"		offX = offXVec[index];			\n" <<
			"		N = NVec[index];				\n" <<
			"		uint task = lid;				\n" <<
			//"		uint count = 0;					\n" <<
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
			//"			if (count < 3) localXB[count*TPB+lid] = xb;	\n" <<
			"			REAL exb = exp(xb);							\n" <<
			"			REAL numer = " << timesX("exb", formatType) << ";\n" <<
			"			REAL denom = (REAL)1.0 + exb;				\n" <<
			"			REAL w = weightVector[vecOffset];\n";
	code << BaseModelG::incrementGradientAndHessianG(formatType, true);
	code << "       	sum0 += gradient; \n" <<
			"       	sum1 += hessian;  \n";
	code << "       	task += TPB; \n" <<
			//"			count += 1;		\n" <<
			"   	} \n";

	code << "		scratch[0][lid] = sum0;	\n" <<
			"		scratch[1][lid] = sum1;	\n";

	code << ReduceBody2<RealType,false>::body();

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
			//"			printf(\"delta %d: %f\\n\", index, delta);	\n" <<
			"			REAL intermediate = max(fabs(delta)*2, bound/2);	\n" <<
			"			intermediate = max(intermediate, 0.001);\n" <<
			"			boundVector[offset] = intermediate;		\n" <<
			"			if (delta != 0) betaVector[offset] = delta + beta;		\n" <<
			"		}										\n";
	code << "   	barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
			"		if (delta != 0) {				\n" <<
			//"			count = 0;							\n" <<
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
			//"				if (count < 3) {				\n" <<
			//"					xb = localXB[count*TPB+lid] + inc; \n" <<
			//"				} else {						\n" <<
			"					xb = xBetaVector[vecOffset] + inc;	\n" <<
			//"				}								\n" <<
			"				xBetaVector[vecOffset] = xb;	\n";
	code << "				task += TPB;					\n";
	//code << "				count += 1;						\n";
	code << "			} 									\n";
	code << "   		barrier(CLK_GLOBAL_MEM_FENCE);      \n";
	code << "		}	\n";
	code << "	}	\n";
	code << "}	\n";
	return SourceCode(code.str(), name);
}

// CV update grad/hess + process delta + update XB
template <class BaseModel, typename WeightType, class BaseModelG>
SourceCode
GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForDoItAllKernel(FormatType formatType, int priorType, bool layoutByPerson) {

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
			"		const uint KStride,				\n" <<
			"		const uint J,					\n" <<
			"		const uint syncCVFolds,			\n" <<
			//"		const uint index)	{				\n";
			"		const uint indexStart,				\n" <<
			"		const uint length,				\n" <<
			"		__global const uint* indices) {   		 	\n";    // TODO Make weight optional
	// Initialization
	code << "	__local uint offK, offX, N, index, allZero;	\n";
	if (priorType == 1) {
		code << "__local REAL lambda;				\n";
		code << "lambda = priorParams[indices[0]];	\n";
	}
	if (priorType == 2) {
		code << "__local REAL var;				\n";
		code << "var = priorParams[indices[0]];	\n";
	}
	code << "	__local REAL grad[TPB];		\n" <<
			"	__local REAL hess[TPB];		\n" <<
			"	__local REAL deltaVec[TPB0];		\n" <<
			"	__local int localDone[TPB0];		\n" <<
			//"	__local int scratchInt[TPB0];		\n" <<
			//"	__local REAL localXB[TPB1*3*TPB0];	\n" <<
			"	uint lid0 = get_local_id(0);			\n" <<
			"	uint lid1 = get_local_id(1);		\n" <<
			"	uint mylid = lid1 * TPB0 + lid0;	\n" <<
			"	uint cvIndex = get_global_id(0);	\n" <<
			"	uint gid1 = get_global_id(1);		\n" <<
			"	uint loopSize = get_global_size(1);	\n" <<
			// "	barrier(CLK_LOCAL_MEM_FENCE);		\n" <<

			"	if (lid1 == 0) {					\n" <<
			"		int temp = doneVector[cvIndex];	\n" <<
			"		localDone[lid0] = temp;	\n" <<
			"	}									\n";

	code << "	if (cvIndex < syncCVFolds) {		\n";
	// code << "	barrier(CLK_LOCAL_MEM_FENCE);		\n";

	code << "	for (int n = 0; n < length; n++) {	\n";

	code << "		index = indices[indexStart + n];	\n" <<
			"		offK = offKVec[index];			\n" <<
			"		offX = offXVec[index];			\n" <<
			"		N = NVec[index];				\n" <<
			"		uint task = gid1;					\n" <<
			//"		uint count = 0;						\n" <<
			"		REAL sum0 = 0.0;					\n" <<
			"		REAL sum1 = 0.0;					\n";


	code <<	"		while (task < N) {		\n";
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
	if (layoutByPerson) {
		code << "	uint vecOffset = k * cvIndexStride + cvIndex;	\n";
	} else {
		code << "	uint vecOffset = k + KStride * cvIndex;			\n";
	}
	code << "		REAL xb = xBetaVector[vecOffset];			\n" <<
			//"		if (count < 3) localXB[(mylid1+myTPB1*count)*myTPB0 + mylid0] = xb;	\n" <<
			"		REAL exb = exp(xb);							\n" <<
			//"		REAL exb = expXBetaVector[vecOffset];	\n" <<
			"		REAL numer = " << timesX("exb", formatType) << ";\n" <<
			"		REAL denom = (REAL)1.0 + exb;				\n" <<
			//"		REAL denom = denomPidVector[vecOffset];		\n" <<
			"		REAL w = weightVector[vecOffset];\n";
	code << BaseModelG::incrementGradientAndHessianG(formatType, true);
	code << "       sum0 += gradient; \n" <<
			"       sum1 += hessian;  \n";
	code << "       task += loopSize; \n" <<
			//"		count += 1;		\n" <<
			"   } \n";

	code << "	grad[mylid] = sum0;	\n" <<
			"	hess[mylid] = sum1;	\n";

	code << "   for(int j = 1; j < TPB1; j <<= 1) {          \n" <<
			"       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
			"       uint mask = (j << 1) - 1;               \n" <<
			"       if ((lid1 & mask) == 0) {                \n" <<
			"           grad[mylid] += grad[mylid + j * TPB0]; \n" <<
			"           hess[mylid] += hess[mylid + j * TPB0]; \n" <<
			"       }                                       \n" <<
			"   }                                         \n";

	code << "	if (lid0 == 0 && lid1 == 0) {			\n" <<
			"		allZero = 1;								\n";
	if (priorType == 1) {
		//code << "	lambda = priorParams[index];				\n";
	}
	if (priorType == 2) {
		//code << "	var = priorParams[index];				\n";
	}
	code << "	}										\n";
	code << "	barrier(CLK_LOCAL_MEM_FENCE);				\n";


	code << "	if (lid1 == 0) {	\n";
//    if (layoutByPerson) {
//		code << "	uint offset = index * cvIndexStride + cvIndex;			\n";
//    } else {
		code << "	uint offset = index + J * cvIndex;			\n";
//    }
    code << "		REAL grad0 = grad[lid0];		\n" <<
			"		grad0 = grad0 - XjYVector[offset];	\n" <<
			"		REAL hess0 = hess[lid0];		\n" <<
			"		REAL beta = betaVector[offset];		\n" <<
			"		REAL delta;							\n";

	if (priorType == 0) {
		code << " 	delta = - grad0 / hess0;			\n";
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

	code << "		delta = delta * localDone[lid0];	\n" <<
			"		REAL bound = boundVector[offset];		\n" <<
			"		if (delta < -bound)	{					\n" <<
			"			delta = -bound;						\n" <<
			"		} else if (delta > bound) {				\n" <<
			"			delta = bound;						\n" <<
			"		}										\n" <<
			"		deltaVec[lid0] = delta;					\n" <<
			"		REAL intermediate = max(fabs(delta)*2, bound/2);	\n" <<
			"		intermediate = max(intermediate, 0.001);	\n" <<
			"		boundVector[offset] = intermediate;		\n" <<
			//"	betaVector[offset] = delta + beta;		\n" <<
			"		if (delta != 0) {						\n" <<
			"			betaVector[offset] = delta + beta;	\n" <<
			"			allZero = 0;						\n" <<
			"		}										\n";
	code << "	}										\n";

	code << "   barrier(CLK_LOCAL_MEM_FENCE);           \n";


	code << "	if (allZero == 0)		{				\n" <<
			//"	if (scratchInt[0] > 0) {				\n" <<
			"		REAL delta = deltaVec[lid0];			\n" <<
			//"		count = 0;							\n" <<
			"		task = lid1;						\n";

	code <<	"		while (task < N) {		\n";
	if (formatType == INDICATOR || formatType == SPARSE) {
		code << "  		uint k = K[offK + task];      	\n";
	} else { // DENSE, INTERCEPT
		code << "   	uint k = task;           		\n";
	}
	if (formatType == SPARSE || formatType == DENSE) {
		code << "   	REAL inc = delta * X[offX + task]; \n";
	} else { // INDICATOR, INTERCEPT
		code << "   	REAL inc = delta;           	\n";
	}
	if (layoutByPerson) {
		code << "		uint vecOffset = k * cvIndexStride + cvIndex;	\n";
	} else {
		code << "		uint vecOffset = k + KStride * cvIndex;			\n";
	}
	code << "			REAL xb;						\n" <<
			//"		if (count < 3) {				\n" <<
			//"			xb = localXB[(mylid1+myTPB1*count)*myTPB0 + mylid0] + inc; \n" <<
			//"		} else {						\n" <<
			"			xb = xBetaVector[vecOffset] + inc;	\n" <<
			//"		}								\n" <<
			//"		REAL xb = xBetaVector[vecOffset] + inc;	\n" <<
			"			xBetaVector[vecOffset] = xb;	\n";
	code << "			task += loopSize;							\n";
			//code << "		count += 1;						\n";
	code << "		} 										\n";

	code << "   	barrier(CLK_GLOBAL_MEM_FENCE);           \n";
	code << "	}	\n";

	code << "}	\n";
	// code << "}	\n";
	code << "}	\n";
	code << "}	\n";
	return SourceCode(code.str(), name);
}

// CV gradient objective
template <class BaseModel, typename RealType, class BaseModelG>
SourceCode
GpuModelSpecifics<BaseModel, RealType, BaseModelG>::writeCodeForGetGradientObjectiveSync(bool isNvidia, bool layoutByPerson) {

	std::string name;
	name = "getGradientObjectiveSync";

	std::stringstream code;
	code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

	code << "__kernel void " << name << "(            \n" <<
			"       const uint N,                     \n" <<
			"       __global const REAL* Y,           \n" <<
			"       __global const REAL* xBetaVector,       \n" <<
			"       __global const REAL* weightVector,	\n" <<
			"       __global REAL* buffer,            \n" <<
			"		const uint cvIndexStride,					\n" <<
			"		const uint KStride) {    \n";    // TODO Make weight optional

	code << "	__local REAL scratch[TPB];				\n";
	code << "	uint lid0 = get_local_id(0);		\n" <<
			"	uint lid1 = get_local_id(1);		\n" <<
			"	uint mylid = lid1 * TPB0 + lid0;		\n";
	code << "	uint cvIndex = get_global_id(0);	\n" <<
			"	uint gid1 = get_global_id(1);		\n" <<
			"	uint loopSize = get_global_size(1);	\n" <<
			"	uint task1 = gid1;					\n" <<
			"	REAL sum = 0.0;						\n";

	code << "	while (task1 < N) {					\n";

	if (layoutByPerson) {
		code << "		uint vecOffset = task1 * cvIndexStride + cvIndex;	\n";
	} else {
		code << "		uint vecOffset = KStride * cvIndex + task1;	\n";
	}
	code << "		REAL w = weightVector[vecOffset];	\n" <<
			"		REAL y = Y[task1];				\n" <<
			"		REAL xb = xBetaVector[vecOffset];	\n" <<
			"		sum += w * y * xb;				\n" <<
			"		task1 += loopSize;		\n" <<
			"	} 									\n";


	if (layoutByPerson) {
		code << "	scratch[mylid] = sum;		\n";
		code << "   for(int j = 1; j < TPB1; j <<= 1) {          \n" <<
	            "       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
	            "       uint mask = (j << 1) - 1;               \n" <<
	            "       if ((lid1 & mask) == 0) {                \n" <<
	            "           scratch[mylid] += scratch[mylid + j * TPB0]; \n" <<
	            "       }                                       \n" <<
	            "   }                                         \n";
	} else {
		code << "	uint lid = lid1;					\n";
		code << "	scratch[lid] = sum;					\n";
		code << (isNvidia ? ReduceBody1<RealType,true>::body() : ReduceBody1<RealType,false>::body());
	}

	code << "	if (lid1 == 0) {					\n" <<
			"		buffer[get_global_id(0) * get_num_groups(1) + get_group_id(1)] = scratch[mylid];	\n" <<
			"	}									\n";

	code << "	}									\n";
	//code << "}  \n"; // End of kernel
	return SourceCode(code.str(), name);
}

// empty kernel to test timings
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




/*
static std::string timesX(const std::string& arg, const FormatType formatType) {
    return (formatType == INDICATOR || formatType == INTERCEPT) ?
        arg : arg + " * x";
}

static std::string weight(const std::string& arg, bool useWeights) {
    return useWeights ? "w * " + arg : arg;
}
*/


} // namespace bsccs

#endif /* KERNELS_HPP_ */
