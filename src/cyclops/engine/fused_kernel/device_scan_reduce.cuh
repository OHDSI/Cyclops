
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::DeviceScanReduce provides device-wide, parallel operations for computing a prefix scan and a tranform-reduction across a sequence of data items residing within device-accessible memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include <cub/config.cuh>
#include <cub/thread/thread_operators.cuh>
#include "dispatch_scan_reduce.cuh"

CUB_NAMESPACE_BEGIN

struct DeviceFuse
{
    /******************************************************************//**
     * \name Inclusive scan and transform-reduction
     *********************************************************************/
    //@{

    /**
     * \brief Computes a device-wide inclusive prefix scan and transform-reduction using the specified
     * binary \p scan_op functor, \p reduction_op functor and \p transform_op functor.
     *
     * @par Snippet
     * The code snippet below illustrates the inclusive prefix scan and transform-reduction of an `int`
     * device vector.
     *
     * @par
     * @code
     * #include <cub/cub.cuh>
     * #include "device_scan_reduce.cuh"
     *
     * // CustomMin functor
     * struct CustomMin
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device-accessible pointers for
     * // input and output
     * int          num_items;      // e.g., 7
     * int          *d_in;          // e.g., [1, 2, 2, 1, 3, 0, 1]
     * int          *d_trans_in;    // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int          *d_sum;         // e.g., [-]
     * CustomMin    min_op;
     * ...
     *
     * // Determine temporary device storage requirements for inclusive
     * // prefix scan
     * void     *d_temp_storage = nullptr;
     * size_t   temp_storage_bytes = 0;
     * DeviceFuse::ScanReduce(
     *   d_temp_storage, temp_storage_bytes,
     *   d_in, d_trans_in, d_sum,
     *   cub::Sum(), min_op, cub::Sum(),
     *   num_items);
     *
     * // Allocate temporary storage for inclusive prefix sum
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run inclusive prefix sum
     * DeviceScan::InclusiveSum(
     *   d_temp_storage, temp_storage_bytes,
     *   d_in, d_trans_in, d_sum,
     *   cub::Sum(), min_op, cub::Sum(),
     *   num_items);
     *
     * // d_sum <-- [26]
     * @endcode
     *
     * @tparam ScanInputIteratorT
     *   **[inferred]** Random-access input iterator type for reading scan
     *   inputs \iterator
     *
     * @tparam TransformInputIteratorT
     *   **[inferred]** Random-access input iterator type for reading additional
     *   input for transformation \iterator
     *
     * @tparam OutputIteratorT
     *   **[inferred]** Random-access output iterator type for writing reduction
     *   outputs \iterator
     *
     * @tparam ScanOpT
     *   **[inferred]** Binary scan functor type having member
     *   `T operator()(const T &a, const T &b)`
     *
     * @tparam ReductionOpT
     *   **[inferred]** Binary reduction functor type having member
     *   `T operator()(const T &a, const T &b)`
     *
     * @tparam TransformOpT
     *   **[inferred]** Binary transform functor type having member
     *   `T operator()(const T &a, const T &b)`
     *
     * @param[in] d_temp_storage
     *   Device-accessible allocation of temporary storage. When `nullptr`, the
     *   required allocation size is written to `temp_storage_bytes` and no
     *   work is done.
     *
     * @param[in,out] temp_storage_bytes
     *   Reference to size in bytes of `d_temp_storage` allocation
     *
     * @param[in] d_in
     *   Random-access iterator to the input sequence of data items
     *
     * @param[in] d_trans_in
     *   Random-access iterator to the additional input sequence of data items
     *
     * @param[out] d_sum
     *   Random-access iterator to the output aggregate
     *
     * @param[in] num_items
     *   Total number of input items (i.e., the length of `d_in`)
     *
     * @param[in] stream
     *   **[optional]** CUDA stream to launch kernels within.
     *   Default is stream<sub>0</sub>.
     *
     * [decoupled look-back]: https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
     */
    template <
        typename        ScanInputIteratorT,
        typename        TransformInputIteratorT,
        typename        OutputIteratorT,
        typename        ScanOpT,
        typename        ReductionOpT,
        typename        TransformOpT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t ScanReduce(
        void                    *d_temp_storage,            ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,        ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        ScanInputIteratorT      d_in,                       ///< [in] Random-access iterator to the input sequence of data items
        TransformInputIteratorT d_trans_in,                 ///< [in] Random-access iterator to the additional input sequence of data items
        OutputIteratorT         d_sum,                      ///< [out] Pointer to the output aggregate
        ScanOpT                 scan_op,                    ///< [in] Binary scan functor
        ReductionOpT            reduction_op,	            ///< [in] Binary reduction functor
        TransformOpT            transform_op,	            ///< [in] Transformation functor on scan output
        int                     num_items,                  ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t            stream             = 0,     ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous  = false) ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchScanReduce<ScanInputIteratorT, TransformInputIteratorT, OutputIteratorT, ScanOpT, ReductionOpT, TransformOpT, NullType, OffsetT>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_trans_in,
            d_sum,
            scan_op,
            reduction_op,
            transform_op,
            NullType(),
            num_items,
            stream,
            debug_synchronous);
    }
    //@}  end member group

};

CUB_NAMESPACE_END


