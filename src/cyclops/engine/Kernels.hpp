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
        if (BaseModel::likelihoodHasDenominator) {
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

/*
	template <class BaseModel, typename WeightType>
    SourceCode
	GpuModelSpecifics<BaseModel, WeightType>::writeCodeForMMGradientHessianKernel(FormatType formatType, bool useWeights, bool isNvidia) {

        std::string name = "computeMMGradHess" + getFormatTypeExtension(formatType) + (useWeights ? "W" : "N");

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
                "       __global const REAL* weight,	  \n" <<
				"		__global const REAL* norm,		  \n" <<
				//"		__global const REAL* fixedBeta,   \n" <<
				"		const uint index) {    \n";    // TODO Make weight optional


        // Initialization
        code << "   const uint lid = get_local_id(0); \n" <<
                "   const uint loopSize = get_global_size(0); \n" <<
                "   uint task = get_global_id(0);  \n" <<
                    // Local and thread storage
#ifdef USE_VECTOR
                "   __local TMP_REAL scratch[TPB]; \n" <<
                "   TMP_REAL sum = 0.0;            \n" <<
#else
                "   __local REAL scratch[2][TPB + 1];  \n" <<
               // "   __local REAL scratch1[TPB];  \n" <<
                "   REAL sum0 = 0.0; \n" <<
                "   REAL sum1 = 0.0; \n" <<

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

        code << "       const REAL exb = expXBeta[k];     	\n" <<
        		"		const REAL xb = xBeta[k];			\n" <<
				"		const REAL norm0 = norm[k];				\n" <<
                "       const REAL numer = " << timesX("exb", formatType) << ";\n" <<
				//"		const REAL denom = denominator[k];	\n";
        		"		const REAL denom = 1 + exb;			\n";
				//"		const REAL factor = norm[k]/abs(x);				\n" <<
        //denominator[k]; \n" <<
                //"       const REAL g = numer / denom;      \n";

        if (useWeights) {
            code << "       const REAL w = weight[k];\n";
        }

        code << "       const REAL gradient = " << weight("numer / denom", useWeights) << ";\n";
        code << "		REAL hessian = 0.0;			\n";


        code << "if (x != 0.0) { \n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       hessian  = " << weight("g  * norm0 / fabs(x) / denom / denom ", useWeights) << ";\n";
        } else {
            code << "       const REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
            		"       hessian  = " << weight("nume2 * norm0 / fabs(x) / denom / denom", useWeights) << ";\n";
        }
        code << "} \n";

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
                "       buffer[get_group_id(0) + 2*get_num_groups(0)*index] = scratch[0][0]; \n" <<
                "       buffer[get_group_id(0) + get_num_groups(0) + 2*get_num_groups(0)*index] = scratch[1][0]; \n" <<
#endif // USE_VECTOR
                "   } \n";


        code << "}  \n"; // End of kernel

        return SourceCode(code.str(), name);
	}


*/
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
				"		__global const int* fixBeta,   \n" <<
				"		const uint indexWorkSize,					\n" <<
				"		const uint wgs,					\n" <<
				"		__global const int* indices) {    \n";    // TODO Make weight optional
        // Initialization
        code << "   const uint lid = get_local_id(0); \n" <<
        		//"   const uint loopSize = get_global_size(0); \n" <<
        		//"   uint task = get_global_id(0);  \n" <<
                "   const uint loopSize = indexWorkSize; \n" <<
				"	const uint index = indices[get_group_id(0)/wgs];		\n" <<
				"	if (fixBeta[index] == 0) {					\n" <<
                "   __local REAL scratch[2][TPB];  \n" <<
                "   uint task = get_global_id(0)%indexWorkSize;  \n" <<
				"	const uint offX = offXVec[index];			\n" <<
				"	const uint offK = offKVec[index];			\n" <<
				"	const uint N = NVec[index];					\n" <<
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

        code << "       const REAL exb = expXBeta[k];     	\n" <<
        		"		const REAL xb = xBeta[k];			\n" <<
				"		const REAL norm0 = norm[k];				\n" <<
                "       const REAL numer = " << timesX("exb", formatType) << ";\n" <<
				"		const REAL denom = denominator[k];	\n";
        		//"		const REAL denom = (REAL)1.0 + exb;			\n";
				//"		const REAL factor = norm[k]/abs(x);				\n" <<
        //denominator[k]; \n" <<
                //"       const REAL g = numer / denom;      \n";

        if (useWeights) {
            code << "       const REAL w = weight[k];\n";
        }

        code << "       const REAL gradient = " << weight("numer / denom", useWeights) << ";\n";
        code << "		REAL hessian = 0.0;			\n";

        if (formatType == INDICATOR || formatType == INTERCEPT) {
            // code << "       hessian  = gradient  * norm0 / denom;				\n";
            code << "       hessian  = " << weight("numer*norm0/denom/denom",useWeights) << ";\n";
        } else {
        	code << "if (x != 0.0) { \n";
            code << "       const REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
            		"       hessian  = " << weight("nume2 * norm0 / fabs(x) / denom / denom", useWeights) << ";\n";
            code << "} \n";
        }

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
				//"	printf(\"%f %f | \", scratch[0][0], scratch[1][0]); \n" <<
#ifdef USE_VECTOR
                "       buffer[get_group_id(0)] = scratch[0]; \n" <<
#else
                //"       buffer[get_group_id(0) + 2*get_num_groups(0)*index] = scratch[0][0]; \n" <<
                //"       buffer[get_group_id(0) + get_num_groups(0) + 2*get_num_groups(0)*index] = scratch[1][0]; \n" <<
				//"       buffer[get_group_id(0)%wgs + 2*wgs*index] = scratch[0][0]; \n" <<
                //"       buffer[get_group_id(0)%wgs + wgs + 2*wgs*index] = scratch[1][0]; \n" <<
				"       buffer[get_group_id(0)%wgs + index*wgs*2] = scratch[0][0]; \n" <<
				"       buffer[get_group_id(0)%wgs + index*wgs*2 + wgs] = scratch[1][0]; \n" <<


#endif // USE_VECTOR
                "   } \n";
        code << "}\n";

        code << "}  \n"; // End of kernel

        return SourceCode(code.str(), name);
	}


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
			"		__global const int* fixBeta,   \n" <<
			"		const uint indexWorkSize,					\n" <<
			"		const uint wgs,					\n" <<
			"		__global const int* indices) {    \n";    // TODO Make weight optional
    // Initialization
    code << "   const uint lid = get_local_id(0); \n" <<
    		//"   const uint loopSize = get_global_size(0); \n" <<
    		//"   uint task = get_global_id(0);  \n" <<
            "   const uint loopSize = indexWorkSize; \n" <<
			"	const uint index = indices[get_group_id(0)/wgs];		\n" <<
			"	if (fixBeta[index] == 0) {					\n" <<
            "   __local REAL scratch[2][TPB];  \n" <<
            "   uint task = get_global_id(0)%indexWorkSize;  \n" <<
			"	const uint offX = offXVec[index];			\n" <<
			"	const uint offK = offKVec[index];			\n" <<
			"	const uint N = NVec[index];					\n" <<
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

    code << "       const REAL exb = expXBeta[k];     	\n" <<
    		"		const REAL xb = xBeta[k];			\n" <<
			"		const REAL norm0 = norm[k];				\n" <<
            "       const REAL numer = " << timesX("exb", formatType) << ";\n" <<
			"		const REAL denom = denominator[k];	\n";
    		//"		const REAL denom = (REAL)1.0 + exb;			\n";
			//"		const REAL factor = norm[k]/abs(x);				\n" <<
    //denominator[k]; \n" <<
            //"       const REAL g = numer / denom;      \n";

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
			//"	printf(\"%f %f | \", scratch[0][0], scratch[1][0]); \n" <<
#ifdef USE_VECTOR
            "       buffer[get_group_id(0)] = scratch[0]; \n" <<
#else
            //"       buffer[get_group_id(0) + 2*get_num_groups(0)*index] = scratch[0][0]; \n" <<
            //"       buffer[get_group_id(0) + get_num_groups(0) + 2*get_num_groups(0)*index] = scratch[1][0]; \n" <<
			//"       buffer[get_group_id(0)%wgs + 2*wgs*index] = scratch[0][0]; \n" <<
            //"       buffer[get_group_id(0)%wgs + wgs + 2*wgs*index] = scratch[1][0]; \n" <<
			"       buffer[get_group_id(0)%wgs + index*wgs*2] = scratch[0][0]; \n" <<
			"       buffer[get_group_id(0)%wgs + index*wgs*2 + wgs] = scratch[1][0]; \n" <<


#endif // USE_VECTOR
            "   } \n";
    code << "}\n";

    code << "}  \n"; // End of kernel

    return SourceCode(code.str(), name);
}


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
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncCVGradientHessianKernel(FormatType formatType, bool useWeights, bool isNvidia) {

        std::string name = "computeSyncCVGradHess" + getFormatTypeExtension(formatType) + (useWeights ? "W" : "N");

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
				"		const uint wgs,					\n" <<
				"		__global const int* cvIndices) {    \n";    // TODO Make weight optional

        // Initialization
        code << "   uint lid = get_local_id(0); \n" <<
                "   uint task = get_global_id(0)%indexWorkSize;  \n" <<
				"	__local uint bufferIndex, cvIndex, vecOffset;	\n" <<
				"	bufferIndex = get_group_id(0)/wgs;	\n" <<
				"	cvIndex = cvIndices[bufferIndex]; \n" <<
				"	vecOffset = stride*cvIndex;	\n" <<
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
        		"		REAL denom = denomPidVector[vecOffset+k];		\n";
        if (useWeights) {
            code << "       REAL w = weightVector[vecOffset+k];\n";
        }

        code << BaseModelG::incrementGradientAndHessianG(formatType, useWeights);

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

	template <class BaseModel, typename WeightType, class BaseModelG>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType, BaseModelG>::writeCodeForSyncComputeRemainingStatisticsKernel() {

        std::string name = "computeRemainingStatistics";

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(     \n" <<
                "       const uint N,              \n" <<
				"		__global REAL* xBetaVector,	   \n" <<
                "       __global REAL* expXBetaVector,   \n" <<
                "       __global REAL* denomPidVector,\n" <<
				"		__global REAL* Y,			\n" <<
				"		__global REAL* Offs,		\n" <<
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
				"		const uint blockSize) {   \n" <<
        "   uint task = get_global_id(0)%blockSize; \n" <<
		"	__local uint cvIndex, vecOffset;	\n" <<
		"	cvIndex = cvIndices[get_global_id(0)/blockSize];	\n" <<
		"	vecOffset = stride*cvIndex;	\n";
        code << "   if (task < N) {      				\n";

        if (formatType == INDICATOR || formatType == SPARSE) {
            code << "   uint k = K[offK + task];         \n";
        } else { // DENSE, INTERCEPT
            code << "   uint k = task;            \n";
        }

        if (formatType == SPARSE || formatType == DENSE) {
            code << "   REAL inc = deltaVector[cvIndex] * X[offX + task]; \n";
        } else { // INDICATOR, INTERCEPT
            code << "   REAL inc = deltaVector[cvIndex];           \n";
        }

        code << "       REAL xb = xBetaVector[vecOffset+k] + inc; 		\n" <<
                "       xBetaVector[vecOffset+k] = xb;                  \n";
        code << "   } \n";
        code << "}    \n";

        return SourceCode(code.str(), name);
    }






} // namespace bsccs

#endif /* KERNELS_HPP_ */
