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


            code << "       const REAL exb = exp(xb); \n" <<
                 "       expXBeta[k] = exb;        \n";
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

} // namespace bsccs

#endif //KERNELSCOX_HPP
