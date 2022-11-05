
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
 * cub::DeviceScanReduce provides device-wide, parallel operations for computing a prefix scan and transform-reduction across a sequence of data items residing within device-accessible memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "agent_scan_reduce.cuh"
#include <cub/thread/thread_operators.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_namespace.cuh>

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Initialization kernel for tile status initialization (multi-block)
 */
template <
    typename            ScanTileStateT>     ///< Tile status interface type
__global__ void DeviceScanReduceInitKernel(
    ScanTileStateT      tile_state,         ///< [in] Tile status interface
    int                 num_tiles)          ///< [in] Number of tiles
{
    // Initialize tile status
    tile_state.InitializeStatus(num_tiles);
}

/**
 * Initialization kernel for tile status initialization (multi-block)
 */
template <
    typename                ScanTileStateT,         ///< Tile status interface type
    typename                NumSelectedIteratorT>   ///< Output iterator type for recording the number of items selected
__global__ void DeviceCompactInitKernel1(
    ScanTileStateT          tile_state,             ///< [in] Tile status interface
    int                     num_tiles,              ///< [in] Number of tiles
    NumSelectedIteratorT    d_num_selected_out)     ///< [out] Pointer to the total number of items selected (i.e., length of \p d_selected_out)
{
    // Initialize tile status
    tile_state.InitializeStatus(num_tiles);

    // Initialize d_num_selected_out
    if ((blockIdx.x == 0) && (threadIdx.x == 0))
        *d_num_selected_out = 0;
}


/**
 * Scan kernel entry point (multi-block)
 */
template <
    typename            ScanPolicyT,        ///< Parameterized ScanPolicyT tuning policy type
    typename            ScanInputIteratorT,     ///< Random-access input iterator type for reading scan inputs \iterator
    typename            TransformInputIteratorT,    ///< Random-access input iterator type for reading additional inputs for transfromation \iterator
    typename            OutputIteratorT,    ///< Random-access output iterator type for writing reduction outputs \iterator
    typename            ScanTileStateT,     ///< Tile status interface type
    typename            ScanOpT,            ///< Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename            ReductionOpT,       ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename            TransformOpT,       ///< Transformation functor on scan output
    typename            InitValueT,         ///< Initial value to seed the exclusive scan (cub::NullType for inclusive scans)
    typename            OffsetT>            ///< Signed integer type for global offsets
__launch_bounds__ (int(ScanPolicyT::BLOCK_THREADS))
__global__ void DeviceScanReduceKernel(
    ScanInputIteratorT      d_in,               ///< Input data
    TransformInputIteratorT     d_trans_in,              ///< Additional input data for transformation
    OutputIteratorT     d_block_sum,        ///< Output data (block reduction)
    OutputIteratorT     d_sum,              ///< Output data (reduction)
    ScanTileStateT      tile_state,         ///< Tile status interface
    int                 start_tile,         ///< The starting tile for the current grid
    ScanOpT             scan_op,            ///< Binary scan functor 
    ReductionOpT        reduction_op,       ///< Binary reduction functor
    TransformOpT        transform_op,       ///< Transformation functor on scan output
    InitValueT          init_value,         ///< Initial value to seed the exclusive scan
    OffsetT             num_items)          ///< Total number of scan items for the entire problem
{
    // Thread block type for scanning input tiles
    typedef AgentScanReduce<
        ScanPolicyT,
        ScanInputIteratorT,
	TransformInputIteratorT,
        OutputIteratorT,
        ScanOpT,
        ReductionOpT,
        TransformOpT,
        InitValueT,
        OffsetT> AgentScanReduceT;

    // Shared memory for AgentScan
    __shared__ typename AgentScanReduceT::TempStorage temp_storage;

    // Process tiles
    AgentScanReduceT(temp_storage, d_in, d_trans_in, d_block_sum, d_sum, scan_op, reduction_op, transform_op, init_value).ConsumeRange(
        num_items,
        tile_state,
        start_tile);


}


/**
 * Reduce a single tile kernel entry point (single-block).  Can be used to aggregate privatized thread block reductions from a previous multi-block reduction pass.
 */
template <
//    typename                ChainedPolicyT,             ///< Chained tuning policy
    typename                SingleTilePolicyT,          ///< Chained tuning policy
    typename                ScanInputIteratorT,             ///< Random-access input iterator type for reading input items \iterator
    typename                OutputIteratorT,            ///< Output iterator type for recording the reduced aggregate \iterator
    typename                OffsetT,                    ///< Signed integer type for global offsets
    typename                ReductionOpT,               ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename                OuputT>                     ///< Data element type that is convertible to the \p value type of \p OutputIteratorT
//__launch_bounds__ (int(ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS), 1)
__launch_bounds__ (int(SingleTilePolicyT::BLOCK_THREADS), 1)
__global__ void DeviceReduceSingleTileKernel1(
    ScanInputIteratorT          d_in,                       ///< [in] Pointer to the input sequence of data items
    OutputIteratorT         d_out,                      ///< [out] Pointer to the output aggregate
    OffsetT                 num_items,                  ///< [in] Total number of input data items
    ReductionOpT            reduction_op,               ///< [in] Binary reduction functor
    OuputT                  init)                       ///< [in] The initial value of the reduction
{
    // Thread block type for reducing input tiles
    typedef AgentReduce<
//	    typename ChainedPolicyT::ActivePolicy::SingleTilePolicy,
            SingleTilePolicyT,
            ScanInputIteratorT,
            OutputIteratorT,
            OffsetT,
            ReductionOpT>
        AgentReduceT;

    // Shared memory storage
    __shared__ typename AgentReduceT::TempStorage temp_storage;

    // Check if empty problem
    if (num_items == 0)
    {
        if (threadIdx.x == 0)
            *d_out = init;
        return;
    }

    // Consume input tiles
    OuputT block_aggregate = AgentReduceT(temp_storage, d_in, reduction_op).ConsumeRange(
        OffsetT(0),
        num_items);

    // Output result
    if (threadIdx.x == 0)
        *d_out = reduction_op(init, block_aggregate);
}

/******************************************************************************
 * Policy
 ******************************************************************************/

template <
    typename OuputT,            ///< Data type
    typename OffsetT,           ///< Signed integer type for global offsets
    typename ReductionOpT>      ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt> 
struct DeviceScanReducePolicy
{
    //------------------------------------------------------------------------------
    // Architecture-specific tuning policies
    //------------------------------------------------------------------------------

    /// SM13
    struct Policy130 : ChainedPolicy<130, Policy130, Policy130>
    {
        // ReducePolicy
        typedef AgentReducePolicy<
                CUB_SCALED_GRANULARITIES(128, 8, OuputT), ///< Threads per block, items per thread
                2,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT>                       ///< Cache load modifier
            ReducePolicy;

        // SingleTilePolicy
        typedef ReducePolicy SingleTilePolicy;

        // SegmentedReducePolicy
        typedef ReducePolicy SegmentedReducePolicy;
    };


    /// SM20
    struct Policy200 : ChainedPolicy<200, Policy200, Policy130>
    {
        // ReducePolicy (GTX 580: 178.9 GB/s @ 48M 4B items, 158.1 GB/s @ 192M 1B items)
        typedef AgentReducePolicy<
                CUB_SCALED_GRANULARITIES(128, 8, OuputT),     ///< Threads per block, items per thread
                4,                                      ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                    ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT>                           ///< Cache load modifier
            ReducePolicy;

        // SingleTilePolicy
        typedef ReducePolicy SingleTilePolicy;

        // SegmentedReducePolicy
        typedef ReducePolicy SegmentedReducePolicy;
    };


    /// SM30
    struct Policy300 : ChainedPolicy<300, Policy300, Policy200>
    {
        // ReducePolicy (GTX670: 154.0 @ 48M 4B items)
        typedef AgentReducePolicy<
                CUB_SCALED_GRANULARITIES(256, 20, OuputT),    ///< Threads per block, items per thread
                2,                                      ///< Number of items per vectorized load
                BLOCK_REDUCE_WARP_REDUCTIONS,           ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT>                           ///< Cache load modifier
            ReducePolicy;

        // SingleTilePolicy
        typedef ReducePolicy SingleTilePolicy;

        // SegmentedReducePolicy
        typedef ReducePolicy SegmentedReducePolicy;
    };


    /// SM35
    struct Policy350 : ChainedPolicy<350, Policy350, Policy300>
    {
        // ReducePolicy (GTX Titan: 255.1 GB/s @ 48M 4B items; 228.7 GB/s @ 192M 1B items)
        typedef AgentReducePolicy<
                CUB_SCALED_GRANULARITIES(256, 20, OuputT),    ///< Threads per block, items per thread
                4,                                      ///< Number of items per vectorized load
                BLOCK_REDUCE_WARP_REDUCTIONS,           ///< Cooperative block-wide reduction algorithm to use
                LOAD_LDG>                               ///< Cache load modifier
            ReducePolicy;

        // SingleTilePolicy
        typedef ReducePolicy SingleTilePolicy;

        // SegmentedReducePolicy
        typedef ReducePolicy SegmentedReducePolicy;
    };

    /// SM60
    struct Policy600 : ChainedPolicy<600, Policy600, Policy350>
    {
        // ReducePolicy (P100: 591 GB/s @ 64M 4B items; 583 GB/s @ 256M 1B items)
        typedef AgentReducePolicy<
                CUB_SCALED_GRANULARITIES(256, 16, OuputT),    ///< Threads per block, items per thread
                4,                                      ///< Number of items per vectorized load
                BLOCK_REDUCE_WARP_REDUCTIONS,           ///< Cooperative block-wide reduction algorithm to use
                LOAD_LDG>                               ///< Cache load modifier
            ReducePolicy;

        // SingleTilePolicy
        typedef ReducePolicy SingleTilePolicy;

        // SegmentedReducePolicy
        typedef ReducePolicy SegmentedReducePolicy;
    };


    /// MaxPolicy
    typedef Policy600 MaxPolicy;

};


/******************************************************************************
 * Dispatch
 ******************************************************************************/


/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceScan
 */
template <
    typename ScanInputIteratorT,     ///< Random-access input iterator type for reading scan inputs \iterator
    typename TransformInputIteratorT,    ///< Random-access input iterator type for reading additional inputs for transformation \iterator
    typename OutputIteratorT,    ///< Random-access output iterator type for writing reduction outputs \iterator
    typename ScanOpT,            ///< Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename ReductionOpT,       ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt> 
    typename TransformOpT,       ///< Transformation functor on scan output    
    typename InitValueT,         ///< The init_value element type for ScanOpT (cub::NullType for inclusive scans)
    typename OffsetT>            ///< Signed integer type for global offsets
struct DispatchScanReduce :
    DeviceReducePolicy<
        typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
            typename std::iterator_traits<ScanInputIteratorT>::value_type,                                  // ... then the input iterator's value type,
            typename std::iterator_traits<OutputIteratorT>::value_type>::Type,                          // ... else the output iterator's value type
        OffsetT,
        ReductionOpT>
{
    //---------------------------------------------------------------------
    // Constants and Types
    //---------------------------------------------------------------------

    enum
    {
        INIT_KERNEL_THREADS = 128
    };

    // The input value type
    typedef typename std::iterator_traits<ScanInputIteratorT>::value_type InputT;

    // The output value type
    typedef typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<ScanInputIteratorT>::value_type,                                          // ... then the input iterator's value type,
        typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type


    // Tile status descriptor interface type
    typedef ScanTileState<InputT> ScanTileStateT;

    int num_tiles;
    int init_grid_size;
    int scan_grid_size;
    int scan_sm_occupancy;
    ScanTileStateT tile_state;

    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------

    /// SM600
    struct Policy600
    {
        typedef AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(128, 15, InputT),      ///< Threads per block, items per thread
                BLOCK_LOAD_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS>
            ScanPolicyT;
    };


    /// SM520
    struct Policy520
    {
        // Titan X: 32.47B items/s @ 48M 32-bit T
        typedef AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(128, 12, InputT),      ///< Threads per block, items per thread
                BLOCK_LOAD_DIRECT,
                LOAD_LDG,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS>
            ScanPolicyT;
    };


    /// SM35
    struct Policy350
    {
        // GTX Titan: 29.5B items/s (232.4 GB/s) @ 48M 32-bit T
        typedef AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(128, 12, InputT),      ///< Threads per block, items per thread
                BLOCK_LOAD_DIRECT,
                LOAD_LDG,
                BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
                BLOCK_SCAN_RAKING>
            ScanPolicyT;
    };

    /// SM30
    struct Policy300
    {
        typedef AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(256, 9, InputT),      ///< Threads per block, items per thread
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS>
            ScanPolicyT;
    };

    /// SM20
    struct Policy200
    {
        // GTX 580: 20.3B items/s (162.3 GB/s) @ 48M 32-bit T
        typedef AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(128, 12, InputT),      ///< Threads per block, items per thread
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS>
            ScanPolicyT;
    };

    /// SM13
    struct Policy130
    {
        typedef AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(96, 21, InputT),      ///< Threads per block, items per thread
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_RAKING_MEMOIZE>
            ScanPolicyT;
    };

    /// SM10
    struct Policy100
    {
        typedef AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(64, 9, InputT),      ///< Threads per block, items per thread
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS>
            ScanPolicyT;
    };


    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

#if (CUB_PTX_ARCH >= 600)
    typedef Policy600 PtxPolicy;

#elif (CUB_PTX_ARCH >= 520)
    typedef Policy520 PtxPolicy;

#elif (CUB_PTX_ARCH >= 350)
    typedef Policy350 PtxPolicy;

#elif (CUB_PTX_ARCH >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_ARCH >= 200)
    typedef Policy200 PtxPolicy;

#elif (CUB_PTX_ARCH >= 130)
    typedef Policy130 PtxPolicy;

#else
    typedef Policy100 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxAgentScanPolicy : PtxPolicy::ScanPolicyT {};
    struct PtxSingleTilePolicy : PtxPolicy::ScanPolicyT {};


    //---------------------------------------------------------------------
    // Utilities
    //---------------------------------------------------------------------

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &scan_kernel_config)
    {
    #if (CUB_PTX_ARCH > 0)
        (void)ptx_version;

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        scan_kernel_config.template Init<PtxAgentScanPolicy>();
        scan_kernel_config.template Init<PtxSingleTilePolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 600)
        {
            scan_kernel_config.template Init<typename Policy600::ScanPolicyT>();
        }
        else if (ptx_version >= 520)
        {
            scan_kernel_config.template Init<typename Policy520::ScanPolicyT>();
        }
        else if (ptx_version >= 350)
        {
            scan_kernel_config.template Init<typename Policy350::ScanPolicyT>();
        }
        else if (ptx_version >= 300)
        {
            scan_kernel_config.template Init<typename Policy300::ScanPolicyT>();
        }
        else if (ptx_version >= 200)
        {
            scan_kernel_config.template Init<typename Policy200::ScanPolicyT>();
        }
        else if (ptx_version >= 130)
        {
            scan_kernel_config.template Init<typename Policy130::ScanPolicyT>();
        }
        else
        {
            scan_kernel_config.template Init<typename Policy100::ScanPolicyT>();
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration.
     */
    struct KernelConfig
    {
        int block_threads;
        int items_per_thread;
        int tile_items;

        template <typename PolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            block_threads       = PolicyT::BLOCK_THREADS;
            items_per_thread    = PolicyT::ITEMS_PER_THREAD;
            tile_items          = block_threads * items_per_thread;
        }
    };


    //---------------------------------------------------------------------
    // Dispatch entrypoints
    //---------------------------------------------------------------------

    /**
     * Internal dispatch routine for computing a device-wide prefix scan using the
     * specified kernel functions.
     */
    template <
//        typename            SingleTilePolicyT,
        typename            ScanReduceInitKernelPtrT,     ///< Function type of cub::DeviceScanInitKernel
        typename            ScanReduceSweepKernelPtrT,    ///< Function type of cub::DeviceScanKernelPtrT
        typename            SingleTileKernelT>            ///< Function type of cub::DeviceReduceSingleTileKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*               d_temp_storage,         ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&             temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        ScanInputIteratorT      d_in,                   ///< [in] Pointer to the input sequence of data items
        TransformInputIteratorT     d_trans_in,                  ///< [in] Pointer to the additional input sequence of data items for transformation
        OutputIteratorT     d_sum,                  ///< [out] Pointer to the output aggregate
        ScanOpT             scan_op,                ///< [in] Binary scan functor 
        ReductionOpT        reduction_op,           ///< [in] Binary reduction functor 	
        TransformOpT        transform_op,           ///< [in] Transformation functor on scan output
        InitValueT          init_value,             ///< [in] Initial value to seed the exclusive scan
        OffsetT             num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t        stream,                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous,      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        int                 /*ptx_version*/,        ///< [in] PTX version of dispatch kernels
        ScanReduceInitKernelPtrT  init_kernel,            ///< [in] Kernel function pointer to parameterization of cub::DeviceScanInitKernel
        ScanReduceSweepKernelPtrT scan_reduce_kernel,            ///< [in] Kernel function pointer to parameterization of cub::DeviceScanKernel
        SingleTileKernelT   single_tile_kernel,     ///< [in] Kernel function pointer to parameterization of cub::DeviceReduceSingleTileKernel
        KernelConfig        scan_kernel_config)     ///< [in] Dispatch parameters that match the policy that \p scan_kernel was compiled for
    {

#ifndef CUB_RUNTIME_ENABLED
        (void)d_temp_storage;
        (void)temp_storage_bytes;
        (void)d_in;
        (void)d_trans_in;        
        (void)d_sum;
        (void)scan_op;
        (void)reduction_op;
        (void)transform_op;
        (void)init_value;
        (void)num_items;
        (void)stream;
        (void)debug_synchronous;
        (void)init_kernel;
        (void)scan_reduce_kernel;
        (void)single_tile_kernel;
        (void)scan_kernel_config;

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

#else
	typedef typename DispatchScanReduce::MaxPolicy MaxPolicyT;
        typedef typename MaxPolicyT::SingleTilePolicy  SingleTilePolicyT;

        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Number of input tiles
            int tile_size = scan_kernel_config.block_threads * scan_kernel_config.items_per_thread;
            int num_tiles = (num_items + tile_size - 1) / tile_size;

            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;;

            // Run grids in epochs (in case number of tiles exceeds max x-dimension
            int scan_grid_size = CUB_MIN(num_tiles, max_dim_x);

            // Specify temporary storage allocation requirements
            size_t  allocation_sizes[2]; // bytes for both scan and reduction
            if (CubDebug(error = ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0]))) break;    // bytes needed for tile status descriptors (scan)
            allocation_sizes[1] =
            {
                    scan_grid_size * sizeof(OutputT)    // bytes needed for privatized block reductions (reduction)
            };

            // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
            void* allocations[2]; // temp storage for both scan and reduction
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
   
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                break;
            }
            OutputT* d_block_sum = (OutputT*) allocations[1];

            // Return if empty problem
            if (num_items == 0)
                break;

            // Construct the tile status interface
            ScanTileStateT tile_state;
            if (CubDebug(error = tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]))) break;

            // Log init_kernel configuration
            int init_grid_size = (num_tiles + INIT_KERNEL_THREADS - 1) / INIT_KERNEL_THREADS;
            if (debug_synchronous) _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);

            // Invoke init_kernel to initialize tile descriptors
            init_kernel<<<init_grid_size, INIT_KERNEL_THREADS, 0, stream>>>(
                tile_state,
                num_tiles);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Get SM occupancy for scan_kernel
            int scan_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                scan_sm_occupancy,            // out
                scan_reduce_kernel,
                scan_kernel_config.block_threads))) break;

            for (int start_tile = 0; start_tile < num_tiles; start_tile += scan_grid_size)
            {
                // Log scan_kernel configuration
                if (debug_synchronous) _CubLog("Invoking %d scan_reduce_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                    start_tile, scan_grid_size, scan_kernel_config.block_threads, (long long) stream, scan_kernel_config.items_per_thread, scan_sm_occupancy);

                // Invoke scan_kernel
                scan_reduce_kernel<<<scan_grid_size, scan_kernel_config.block_threads, 0, stream>>>(
                    d_in,
                    d_trans_in,
                    d_block_sum,
                    d_sum,
                    tile_state,
                    start_tile,
                    scan_op,
                    reduction_op,
                    transform_op,
                    init_value,
                    num_items);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
            }
	    
            // Log single_reduce_sweep_kernel configuration
            if (debug_synchronous) _CubLog("Invoking DeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), %d items per thread\n",
                SingleTilePolicyT::BLOCK_THREADS,
                (long long) stream,
                SingleTilePolicyT::ITEMS_PER_THREAD);

            // Invoke DeviceReduceSingleTileKernel
            single_tile_kernel<<<1, SingleTilePolicyT::BLOCK_THREADS, 0, stream>>>(
                d_block_sum,
                d_sum,
                scan_grid_size,
                reduction_op,
                OutputT());

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

        }
        while (0);

        return error;

#endif  // CUB_RUNTIME_ENABLED
    }

     /**
     * Internal dispatch routine
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*           d_temp_storage,         ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&         temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        ScanInputIteratorT  d_in,                   ///< [in] Pointer to the input sequence of data items
        TransformInputIteratorT d_trans_in,                  ///< [in] Pointer to the additional input sequence of data items for transformation
        OutputIteratorT d_sum,                  ///< [out] Pointer to the output aggregate
        ScanOpT         scan_op,                ///< [in] Binary scan functor 
        ReductionOpT    reduction_op,           ///< [in] Binary reduction functor
        TransformOpT    transform_op,           ///< [in] Transformation functor on scan output	
        InitValueT      init_value,             ///< [in] Initial value to seed the exclusive scan
        OffsetT         num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t    stream,                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous)      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        typedef typename DispatchScanReduce::MaxPolicy MaxPolicyT;
        typedef typename MaxPolicyT::SingleTilePolicy  SingleTilePolicyT;

        cudaError error = cudaSuccess;
//        debug_synchronous = true;
        do
        {
            // Get PTX version
            int ptx_version;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Get kernel kernel dispatch configurations
            KernelConfig scan_kernel_config;
            InitConfigs(ptx_version, scan_kernel_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_trans_in,
                d_sum,
                scan_op,
                reduction_op,
                transform_op,
                init_value,
                num_items,
                stream,
                debug_synchronous,
                ptx_version,
                DeviceScanReduceInitKernel<ScanTileStateT>,
                DeviceScanReduceKernel<PtxAgentScanPolicy, ScanInputIteratorT, TransformInputIteratorT, OutputIteratorT, ScanTileStateT, ScanOpT, ReductionOpT, TransformOpT, InitValueT, OffsetT>,
                DeviceReduceSingleTileKernel1<SingleTilePolicyT, OutputIteratorT, OutputIteratorT, OffsetT, ReductionOpT, OutputT>,
                scan_kernel_config))) break;
        }
        while (0);

        return error;
    }
};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


