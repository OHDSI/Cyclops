//
// Created by Jianxiao Yang on 2019-12-30.
//

#ifndef KERNELSCOX_HPP
#define KERNELSCOX_HPP

#include <boost/compute/type_traits/type_name.hpp>
#include "BaseKernels.hpp"
#include "ModelSpecifics.h"

namespace bsccs{

    template <class BaseModel, typename RealType>
    SourceCode
    GpuModelSpecificsCox<BaseModel, RealType>::writeCodeForUpdateXBetaKernel(FormatType formatType) {

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
                "       __global const int* id) {  \n" <<
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

        code << "   if (task < N) {      \n";
        code << "       REAL xb = xBeta[k] + inc; \n" <<
                "       xBeta[k] = xb;                  \n";

        if (BaseModel::likelihoodHasDenominator) {
            // TODO: The following is not YET thread-safe for multi-row observations
             code << "       const REAL oldEntry = expXBeta[k];                 \n" <<
                     "       const REAL newEntry = expXBeta[k] = exp(xBeta[k]); \n" <<
                     "       denominator[" << group<BaseModel>("id","k") << "] += (newEntry - oldEntry); \n";


//            code << "       const REAL exb = exp(xb); \n" <<
//                    "       expXBeta[k] = exb;        \n";
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

//    template <class BaseModel, typename RealType>
//    SourceCode
//    GpuModelSpecifics<BaseModel, RealType>::writeCodeForGradientHessianKernel(FormatType formatType, bool useWeights, bool isNvidia) {
//
//        std::string name = "computeGradHess" + getFormatTypeExtension(formatType) + (useWeights ? "W" : "N");
//
//        std::stringstream code;
//        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
//
//        code << "__kernel void " << name << "(            \n" <<
//             "       const uint offX,                  \n" <<
//             "       const uint offK,                  \n" <<
//             "       const uint N,                     \n" <<
//             "       __global const REAL* X,           \n" <<
//             "       __global const int* K,            \n" <<
//             "       __global const REAL* Y,           \n" <<
//             "       __global const REAL* xBeta,       \n" <<
//             "       __global const REAL* expXBeta,    \n" <<
//             "       __global const REAL* denominator, \n" <<
//             #ifdef USE_VECTOR
//             "       __global TMP_REAL* buffer,     \n" <<
//             #else
//             "       __global REAL* buffer,            \n" <<
//             #endif // USE_VECTOR
//             "       __global const int* id,           \n" <<  // TODO Make id optional
//             "       __global const REAL* weight) {    \n";    // TODO Make weight optional
//
//        // Initialization
//        code << "   const uint lid = get_local_id(0); \n" <<
//                "   const uint loopSize = get_global_size(0); \n" <<
//                "   uint task = get_global_id(0);  \n" <<
//             // Local and thread storage
//             #ifdef USE_VECTOR
//             "   __local TMP_REAL scratch[TPB]; \n" <<
//                "   TMP_REAL sum = 0.0;            \n" <<
//             #else
//             "   __local REAL scratch[2][TPB + 1];  \n" <<
//             // "   __local REAL scratch1[TPB];  \n" <<
//             "   REAL sum0 = 0.0; \n" <<
//             "   REAL sum1 = 0.0; \n" <<
//             #endif // USE_VECTOR
//             //
//             "   while (task < N) { \n";
//
//        // Fused transformation-reduction
//
//        if (formatType == INDICATOR || formatType == SPARSE) {
//            code << "       const uint k = K[offK + task];         \n";
//        } else { // DENSE, INTERCEPT
//            code << "       const uint k = task;            \n";
//        }
//
//        if (formatType == SPARSE || formatType == DENSE) {
//            code << "       const REAL x = X[offX + task]; \n";
//        } else { // INDICATOR, INTERCEPT
//            // Do nothing
//        }
//
//        code << "       const REAL exb = expXBeta[k];     \n" <<
//                "       const REAL numer = " << timesX("exb", formatType) << ";\n" <<
//                "       const REAL denom = 1.0 + exb;      \n" <<
//             //denominator[k]; \n" <<
//                "       const REAL g = numer / denom;      \n";
//
//        if (useWeights) {
//            code << "       const REAL w = weight[k];\n";
//        }
//
//        code << "       const REAL gradient = " << weight("g", useWeights) << ";\n";
//        if (formatType == INDICATOR || formatType == INTERCEPT) {
//            code << "       const REAL hessian  = " << weight("g * ((REAL)1.0 - g)", useWeights) << ";\n";
//        } else {
//            code << "       const REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
//                    "       const REAL hessian  = " << weight("(nume2 / denom - g * g)", useWeights) << ";\n";
//        }
//
//#ifdef USE_VECTOR
//        code << "       sum += (TMP_REAL)(gradient, hessian); \n";
//#else
//        code << "       sum0 += gradient; \n" <<
//                "       sum1 += hessian;  \n";
//#endif // USE_VECTOR
//
//        // Bookkeeping
//        code << "       task += loopSize; \n" <<
//             "   } \n" <<
//             // Thread -> local
//             #ifdef USE_VECTOR
//             "   scratch[lid] = sum; \n";
//             #else
//             "   scratch[0][lid] = sum0; \n" <<
//             "   scratch[1][lid] = sum1; \n";
//#endif // USE_VECTOR
//
//#ifdef USE_VECTOR
//        // code << (isNvidia ? ReduceBody1<real,true>::body() : ReduceBody1<real,false>::body());
//        code << ReduceBody1<real,false>::body();
//#else
//        code << (isNvidia ? ReduceBody2<real,true>::body() : ReduceBody2<real,false>::body());
//#endif
//
//        code << "   if (lid == 0) { \n" <<
//             #ifdef USE_VECTOR
//             "       buffer[get_group_id(0)] = scratch[0]; \n" <<
//             #else
//             "       buffer[get_group_id(0)] = scratch[0][0]; \n" <<
//             "       buffer[get_group_id(0) + get_num_groups(0)] = scratch[1][0]; \n" <<
//             #endif // USE_VECTOR
//             "   } \n";
//
//        code << "}  \n"; // End of kernel
//
//        return SourceCode(code.str(), name);
//    }

} // namespace bsccs

#endif //KERNELSCOX_HPP
