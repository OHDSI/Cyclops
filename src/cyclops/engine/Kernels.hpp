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

template<typename T, bool isNvidiaDevice>
struct ReduceBody4 {
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
			"			scratch[2][lid] += scratch[2][lid + j];	\n" <<
			"			scratch[3][lid] += scratch[3][lid + j];	\n" <<
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

template<typename T>
struct ReduceBody4<T,true>
{
    static std::string body() {
        std::stringstream k;
        k <<
            "   barrier(CLK_LOCAL_MEM_FENCE); \n" <<
            "   if (TPB >= 1024) { if (lid < 512) { sum0 += scratch[0][lid + 512]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 512]; scratch[1][lid] = sum1; \n" <<
			"										sum2 += scratch[2][lid + 512]; scratch[2][lid] = sum2; sum3 += scratch[3][lid + 512]; scratch[3][lid] = sum3; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
            "   if (TPB >=  512) { if (lid < 256) { sum0 += scratch[0][lid + 256]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 256]; scratch[1][lid] = sum1; \n" <<
			"										sum2 += scratch[2][lid + 256]; scratch[2][lid] = sum2; sum3 += scratch[3][lid + 256]; scratch[3][lid] = sum3; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
            "   if (TPB >=  256) { if (lid < 128) { sum0 += scratch[0][lid + 128]; scratch[0][lid] = sum0; sum1 += scratch[1][lid + 128]; scratch[1][lid] = sum1; \n" <<
			"										sum2 += scratch[2][lid + 128]; scratch[2][lid] = sum2; sum3 += scratch[3][lid + 128]; scratch[3][lid] = sum3; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
            "   if (TPB >=  128) { if (lid <  64) { sum0 += scratch[0][lid +  64]; scratch[0][lid] = sum0; sum1 += scratch[1][lid +  64]; scratch[1][lid] = sum1; \n" <<
			"										sum2 += scratch[2][lid +  64]; scratch[2][lid] = sum2; sum3 += scratch[3][lid +  64]; scratch[3][lid] = sum3; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
        // warp reduction
            "   if (lid < 32) { \n" <<
        // volatile this way we don't need any barrier
            "       volatile __local TMP_REAL **lmem = scratch; \n" <<
            "       if (TPB >= 64) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+32]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+32]; lmem[2][lid] = sum2 = sum2 + lmem[2][lid+32]; lmem[3][lid] = sum3 = sum3 + lmem[3][lid+32];} \n" <<
            "       if (TPB >= 32) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+16]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+16]; lmem[2][lid] = sum2 = sum2 + lmem[2][lid+16]; lmem[3][lid] = sum3 = sum3 + lmem[3][lid+16];} \n" <<
            "       if (TPB >= 16) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 8]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 8]; lmem[2][lid] = sum2 = sum2 + lmem[2][lid+ 8]; lmem[3][lid] = sum3 = sum3 + lmem[3][lid+ 8];} \n" <<
            "       if (TPB >=  8) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 4]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 4]; lmem[2][lid] = sum2 = sum2 + lmem[2][lid+ 4]; lmem[3][lid] = sum3 = sum3 + lmem[3][lid+ 4];} \n" <<
            "       if (TPB >=  4) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 2]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 2]; lmem[2][lid] = sum2 = sum2 + lmem[2][lid+ 2]; lmem[3][lid] = sum3 = sum3 + lmem[3][lid+ 2];} \n" <<
            "       if (TPB >=  2) { lmem[0][lid] = sum0 = sum0 + lmem[0][lid+ 1]; lmem[1][lid] = sum1 = sum1 + lmem[1][lid+ 1]; lmem[2][lid] = sum2 = sum2 + lmem[2][lid+ 1]; lmem[3][lid] = sum3 = sum3 + lmem[3][lid+ 1];} \n" <<
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
}; // anonymous namespace

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
                "       __global REAL* buffer,            \n" <<
                "       __global const int* id,           \n" <<  // TODO Make id optional
                "       __global const REAL* weight) {    \n";    // TODO Make weight optional

        // Initialization
        /*
        code << "   const uint lid = get_local_id(0); \n" <<
                "   const uint loopSize = get_global_size(0); \n" <<
                "   uint task = get_global_id(0);  \n" <<
                    // Local and thread storage
                "   __local REAL scratch[2][TPB];  \n" <<
               // "   __local REAL scratch1[TPB];  \n" <<
                "   REAL sum0 = 0.0; \n" <<
                "   REAL sum1 = 0.0; \n";
        */
        code << "	REAL a = exp(0.0);	\n";
                    //
/*
        code << "   while (task < N) { \n";

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
        		//"		REAL exb = exp(0.0);		\n" <<
        		"		REAL exb = " << BaseModelG::getOffsExpXBetaG("0","xb") << ";\n" <<
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

        code << "       sum0 += gradient; \n" <<
                "       sum1 += hessian;  \n";

        // Bookkeeping
        code << "       task += loopSize; \n";

               code <<  "   } \n";

/*

                    // Thread -> local

                "   scratch[0][lid] = sum0; \n" <<
                "   scratch[1][lid] = sum1; \n";

        code << (isNvidia ? ReduceBody2<RealType,true>::body() : ReduceBody2<RealType,false>::body());

        code << "   if (lid == 0) { \n" <<

                "       buffer[get_group_id(0)] = scratch[0][0]; \n" <<
                "       buffer[get_group_id(0) + get_num_groups(0)] = scratch[1][0]; \n" <<
                "   } \n";

                */

        code << "}  \n"; // End of kernel

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
			    	code << "		uint vecOffset = k + cvIndexStride * cvIndex;	\n";
			    }

				code << "			sum += " << timesX("Y[k] * weightVector[vecOffset]", formatType) << ";\n";

			    code << "			task += TPB;						\n";
			    code << "		}										\n";

			    code << "		scratch[lid] = sum;						\n";

		        code << ReduceBody1<RealType,false>::body();

		        code << "		if (lid == 0) {							\n";
		        if (layoutByPerson) {
		        	code << "		uint vecOffset = index * cvIndexStride + cvIndex; \n";
		        } else {
		        	code << "		uint vecOffset = cvIndex * J + index;			\n";
		        }
		        code << "			XjYVector[vecOffset] = scratch[lid];		\n";
		        code << "		}										\n";

		        code << "		barrier(CLK_GLOBAL_MEM_FENCE);			\n";

		        code << "	}										\n";


			    code << "	}										\n";


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
