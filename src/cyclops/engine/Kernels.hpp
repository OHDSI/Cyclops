/*
 * Kernels.hpp
 *
 *  Created on: Apr 4, 2016
 *      Author: msuchard
 */

#ifndef KERNELS_HPP_
#define KERNELS_HPP_

#include <boost/compute/type_traits/type_name.hpp>

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
            "           scratch0[lid] += scratch0[lid + j]; \n" <<
            "           scratch1[lid] += scratch1[lid + j]; \n" <<
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
            "       volatile __local TMP_REAL *lmem = scratch;                  \n" <<
            "       if (TPB >= 64) { lmem[lid] = sum = sum + lmem[lid+32]; } \n" <<
            "       if (TPB >= 32) { lmem[lid] = sum = sum + lmem[lid+16]; } \n" <<
            "       if (TPB >= 16) { lmem[lid] = sum = sum + lmem[lid+ 8]; } \n" <<
            "       if (TPB >=  8) { lmem[lid] = sum = sum + lmem[lid+ 4]; } \n" <<
            "       if (TPB >=  4) { lmem[lid] = sum = sum + lmem[lid+ 2]; } \n" <<
            "       if (TPB >=  2) { lmem[lid] = sum = sum + lmem[lid+ 1]; } \n" <<
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
            "   if (TPB >= 1024) { if (lid < 512) { sum0 += scratch0[lid + 512]; scratch0[lid] = sum0; sum1 += scratch1[lid + 512]; scratch1[lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
            "   if (TPB >=  512) { if (lid < 256) { sum0 += scratch0[lid + 256]; scratch0[lid] = sum0; sum1 += scratch1[lid + 256]; scratch1[lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
            "   if (TPB >=  256) { if (lid < 128) { sum0 += scratch0[lid + 128]; scratch0[lid] = sum0; sum1 += scratch1[lid + 128]; scratch1[lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
            "   if (TPB >=  128) { if (lid <  64) { sum0 += scratch0[lid +  64]; scratch0[lid] = sum0; sum1 += scratch1[lid +  64]; scratch1[lid] = sum1; } barrier(CLK_LOCAL_MEM_FENCE); } \n" <<
        // warp reduction
            "   if (lid < 32) { \n" <<
        // volatile this way we don't need any barrier
            "       volatile __local TMP_REAL *lmem0 = scratch0; \n" <<
            "       volatile __local TMP_REAL *lmem1 = scratch1; \n" <<
            "       if (TPB >= 64) { lmem0[lid] = sum0 = sum0 + lmem0[lid+32]; lmem1[lid] = sum1 = sum1 + lmem1[lid+32]; } \n" <<
            "       if (TPB >= 32) { lmem0[lid] = sum0 = sum0 + lmem0[lid+16]; lmem1[lid] = sum1 = sum1 + lmem1[lid+16]; } \n" <<
            "       if (TPB >= 16) { lmem0[lid] = sum0 = sum0 + lmem0[lid+ 8]; lmem1[lid] = sum1 = sum1 + lmem1[lid+ 8]; } \n" <<
            "       if (TPB >=  8) { lmem0[lid] = sum0 = sum0 + lmem0[lid+ 4]; lmem1[lid] = sum1 = sum1 + lmem1[lid+ 4]; } \n" <<
            "       if (TPB >=  4) { lmem0[lid] = sum0 = sum0 + lmem0[lid+ 2]; lmem1[lid] = sum1 = sum1 + lmem1[lid+ 2]; } \n" <<
            "       if (TPB >=  2) { lmem0[lid] = sum0 = sum0 + lmem0[lid+ 1]; lmem1[lid] = sum1 = sum1 + lmem1[lid+ 1]; } \n" <<
            "   }                                            \n";
        return k.str();
    }
};

    static std::string timesX(const std::string arg, const FormatType formatType) {
        return (formatType == INDICATOR || formatType == INTERCEPT) ?
            arg : arg + " * x";
    }

    static std::string weight(const std::string arg, bool useWeights) {
        return useWeights ? "w * " + arg : arg;
    }

}; // anonymous namespace

	template <class BaseModel, typename WeightType>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType>::writeCodeForGradientHessianKernel(FormatType formatType, bool useWeights) {

        std::string name = "computeGradHess" + getFormatTypeExtension(formatType) + (useWeights ? "W" : "N");

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

//         code << "#define GROUP(id, k) " <<
//             (BaseModel::hasIndependentRows ? "k" : "id[k]") << "\n";

        code << "__kernel void " << name << "(            \n" <<
                "       __global REAL* X,                 \n" <<
                "       __global const int* K,            \n" <<
                "       const uint N,                     \n" <<
                "       const REAL beta,                  \n" << // TODO Remove
                "       __global const REAL* Y,           \n" <<
                "       __global const REAL* xBeta,       \n" <<
                "       __global const REAL* expXBeta,    \n" <<
                "       __global const REAL* denominator, \n" <<
                "       __global REAL* buffer,            \n" <<
                "       __global const int* id,           \n" <<  // TODO Make id optional
                "       __global const REAL* weight) {    \n";    // TODO Make weight optional

        // Initialization
        code << "   const uint lid = get_local_id(0); \n" <<
                "   const uint loopSize = get_global_size(0); \n" <<
                "   uint task = get_global_id(0);  \n" <<
                    // Local storage
                "   __local REAL scratch0[TPB];  \n" <<
                "   __local REAL scratch1[TPB];  \n" <<
                    // Thread storage
                "   REAL sum0 = 0.0; \n" <<
                "   REAL sum1 = 0.0; \n" <<
                    //
                "   while (task < N) { \n";

        // Fused transformation-reduction

        if (formatType == INDICATOR || formatType == SPARSE) {
            code << "       const uint k = K[task];         \n";
        } else { // DENSE, INTERCEPT
            code << "       const uint k = task;            \n";
        }

        if (formatType == SPARSE || formatType == DENSE) {
            code << "       const REAL x = X[task]; \n";
        } else { // INDICATOR, INTERCEPT
            // Do nothing
        }

        code << "       const REAL numer = " << timesX("expXBeta[k]", formatType) << ";\n" <<
                "       const REAL denom = denominator[k]; \n" <<
                "       const REAL g = numer / denom;      \n";

        if (useWeights) {
            code << "       const REAL w = weight[k];\n";
        }

        code << "       const REAL gradient = " << weight("g", useWeights) << ";\n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       const REAL hessian  = " << weight("g * ((REAL)1.0 - g)", useWeights) << ";\n";
        } else {
            code << "       const REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
                    "       const REAL hessian  = " << weight("(nume2 / denom - g * g)", useWeights) << ";\n";
        }

        code << "       sum0 += gradient; \n" <<
                "       sum1 += hessian;  \n";

        // Bookkeeping
        code << "       task += loopSize; \n" <<
                "   } \n" <<
                    // Thread -> local
                "   scratch0[lid] = sum0; \n" <<
                "   scratch1[lid] = sum1; \n";

        code << ReduceBody2<real,false>::body();

        code << "   if (lid == 0) { \n" <<
                "       buffer[get_group_id(0)] = scratch0[0]; \n" <<
                "       buffer[get_group_id(0) + get_num_groups(0)] = scratch1[0]; \n" <<
                "   } \n";

        code << "}  \n"; // End of kernel

        return SourceCode(code.str(), name);
    }

	template <class BaseModel, typename WeightType>
    SourceCode
    GpuModelSpecifics<BaseModel, WeightType>::writeCodeForUpdateXBetaKernel(FormatType formatType) {

        std::string name = "updateXBeta" + getFormatTypeExtension(formatType);

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        if (BaseModel::hasIndependentRows) {
            code << "#define GROUP(id, k) k     \n";
        } else {
            code << "#define GROUP(id, k) id[k] \n";
        }

        code << "__kernel void " << name << "(     \n" <<
                "       __global const REAL* X,    \n" <<
                "       __global const int* K,     \n" <<
                "       const uint N,              \n" <<
                "       const REAL delta,          \n" <<
                "       __global const REAL* Y,    \n" <<
                "       __global REAL* xBeta,      \n" <<
                "       __global REAL* expXBeta,   \n" <<
                "       __global REAL* denominator,\n" <<
                "       __global const int* id) {  \n" <<
                "   const uint task = get_global_id(0); \n";

        if (formatType == INDICATOR || formatType == SPARSE) {
            code << "   const uint k = K[task];         \n";
        } else { // DENSE, INTERCEPT
            code << "   const uint k = task;            \n";
        }

        if (formatType == SPARSE || formatType == DENSE) {
            code << "   const REAL inc = delta * X[task]; \n";
        } else { // INDICATOR, INTERCEPT
            code << "   const REAL inc = delta;           \n";
        }

        code << "   if (task < N) {      \n" <<
                "       xBeta[k] += inc; \n";

        if (BaseModel::likelihoodHasDenominator) {
            // TODO: The following is not YET thread-safe for multi-row observations
            code << "       const REAL oldEntry = expXBeta[k];                 \n" <<
                    "       const REAL newEntry = expXBeta[k] = exp(xBeta[k]); \n" <<
                    "       denominator[GROUP(id,k)] += (newEntry - oldEntry); \n";

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


} // namespace bsccs

#endif /* KERNELS_HPP_ */
