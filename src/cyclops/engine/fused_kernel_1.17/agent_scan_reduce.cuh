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
 * cub::AgentScanReduce implements a stateful abstraction of CUDA thread blocks for participating in device-wide prefix scan and transform-reduction.
 */

#pragma once

#include <iterator>

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include "block_store1.cuh"
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/config.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

CUB_NAMESPACE_BEGIN


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentScan
 */
template <
    int                         NOMINAL_BLOCK_THREADS_4B,       ///< Threads per thread block
    int                         NOMINAL_ITEMS_PER_THREAD_4B,    ///< Items per thread (per tile of input)
    typename                    ComputeT,                       ///< Dominant compute type
    int                         _VECTOR_LOAD_LENGTH,            ///< Number of items per vectorized load
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    BlockStoreAlgorithm1         _STORE_ALGORITHM,               ///< The BlockStore algorithm to use
    BlockScanAlgorithm          _SCAN_ALGORITHM,                ///< The BlockScan algorithm to use
    BlockReduceAlgorithm        _BLOCK_ALGORITHM,               ///< Cooperative block-wide reduction algorithm to use
    typename                    ScalingType =  MemBoundScaling<NOMINAL_BLOCK_THREADS_4B, NOMINAL_ITEMS_PER_THREAD_4B, ComputeT> >

struct AgentScanReducePolicy :
    ScalingType
{
    enum
    {
        VECTOR_LOAD_LENGTH  = _VECTOR_LOAD_LENGTH,  ///< Number of items per vectorized load
    };
 
    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;          ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;           ///< Cache load modifier for reading input elements
    static const BlockStoreAlgorithm1    STORE_ALGORITHM         = _STORE_ALGORITHM;         ///< The BlockStore algorithm to use
    static const BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;          ///< The BlockScan algorithm to use
    static const BlockReduceAlgorithm   BLOCK_ALGORITHM         = _BLOCK_ALGORITHM;         ///< Cooperative block-wide reduction algorithm to use
};




/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief AgentScan implements a stateful abstraction of CUDA thread blocks for participating in device-wide prefix scan .
 */
template <
    typename AgentScanReducePolicyT,    ///< Parameterized AgentScanReducePolicyT tuning policy type
    typename ScanInputIteratorT,        ///< Random-access input iterator type
    typename TransformInputIteratorT,   ///< Random-access input iterator type
    typename OutputIteratorT,           ///< Random-access iterator type for output
    typename ScanOpT,                   ///< Scan functor type
    typename ReductionOpT,              ///< Reduction functor type
    typename TransformOpT,              ///< Transformation functor type
    typename InitValueT,                ///< The init_value element for ScanOpT type (cub::NullType for inclusive scan)
    typename OffsetT>                   ///< Signed integer type for global offsets
struct AgentScanReduce
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // The input value type
    using ScanInputT = cub::detail::value_t<ScanInputIteratorT>;
    using TransformInputT = cub::detail::value_t<TransformInputIteratorT>;

    // The output value type -- used as the intermediate accumulator
    // Per https://wg21.link/P0571, use InitValueT if provided, otherwise the
    // input iterator's value type.
    using ScanOutputT =
      cub::detail::conditional_t<std::is_same<InitValueT, NullType>::value,
                                 ScanInputT,
                                 InitValueT>;
    using ReduceOutputT = cub::detail::non_void_value_t<OutputIteratorT, ScanInputT>;

    // Tile status descriptor interface type
    using ScanTileStateT = ScanTileState<ScanOutputT>;

    // Input iterator wrapper type (for applying cache modifier)
    // Wrap the native input pointer with CacheModifiedInputIterator
    // or directly use the supplied input iterator type
    using WrappedScanInputIteratorT = cub::detail::conditional_t<
      std::is_pointer<ScanInputIteratorT>::value,
      CacheModifiedInputIterator<AgentScanReducePolicyT::LOAD_MODIFIER, ScanInputT, OffsetT>,
      ScanInputIteratorT>;
    using WrappedTransformInputIteratorT = cub::detail::conditional_t<
      std::is_pointer<TransformInputIteratorT>::value,
      CacheModifiedInputIterator<AgentScanReducePolicyT::LOAD_MODIFIER, TransformInputT, OffsetT>,
      TransformInputIteratorT>;

    // Constants
    enum
    {
        // Inclusive scan if no init_value type is provided
        IS_INCLUSIVE        = std::is_same<InitValueT, NullType>::value,
        BLOCK_THREADS       = AgentScanReducePolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD    = AgentScanReducePolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Parameterized BlockLoad type
    typedef BlockLoad<
            ScanInputT,
            AgentScanReducePolicyT::BLOCK_THREADS,
            AgentScanReducePolicyT::ITEMS_PER_THREAD,
            AgentScanReducePolicyT::LOAD_ALGORITHM>
        BlockLoadScanT;
    typedef BlockLoad<
            TransformInputT,
            AgentScanReducePolicyT::BLOCK_THREADS,
            AgentScanReducePolicyT::ITEMS_PER_THREAD,
            AgentScanReducePolicyT::LOAD_ALGORITHM>
        BlockLoadTransformT;

    // Parameterized BlockStore type
    typedef BlockStore1<
            ScanInputT,
            AgentScanReducePolicyT::BLOCK_THREADS,
            AgentScanReducePolicyT::ITEMS_PER_THREAD,
            AgentScanReducePolicyT::STORE_ALGORITHM>
        BlockStoreScanT;
    typedef BlockStore1<
            TransformInputT,
            AgentScanReducePolicyT::BLOCK_THREADS,
            AgentScanReducePolicyT::ITEMS_PER_THREAD,
            AgentScanReducePolicyT::STORE_ALGORITHM>
        BlockStoreTransformT;

    // Parameterized BlockScan type
    typedef BlockScan<
            ScanInputT,
            AgentScanReducePolicyT::BLOCK_THREADS,
            AgentScanReducePolicyT::SCAN_ALGORITHM>
        BlockScanT;

    // Parameterized BlockReduce primitive

    typedef BlockReduce<
            ReduceOutputT,
            AgentScanReducePolicyT::BLOCK_THREADS,
            BLOCK_REDUCE_WARP_REDUCTIONS> // AgentScanReducePolicyT::BLOCK_ALGORITHM
        BlockReduceT;

    // Callback type for obtaining tile prefix during block scan
    typedef TilePrefixCallbackOp<
            ScanInputT,
            ScanOpT,
            ScanTileStateT>
        TilePrefixCallbackOpT;

    // Stateful BlockScan prefix callback type for managing a running total while scanning consecutive tiles
    typedef BlockScanRunningPrefixOp<
            ScanInputT,
            ScanOpT>
        RunningPrefixCallbackOp;

    // Shared memory type for this thread block
    union _TempStorage
    {
        typename BlockLoadScanT::TempStorage    load;       // Smem needed for tile loading
        typename BlockStoreScanT::TempStorage   store;      // Smem needed for tile storing
        typename BlockLoadTransformT::TempStorage    trans_load;       // Smem needed for tile loading
        typename BlockStoreTransformT::TempStorage   trans_store;      // Smem needed for tile storing

        struct ScanReduceStorage
        {
            typename TilePrefixCallbackOpT::TempStorage  prefix;     // Smem needed for cooperative prefix callback
            typename BlockScanT::TempStorage             scan;       // Smem needed for tile scanning
	    typename BlockReduceT::TempStorage           reduce;
        } scan_reduce_storage;
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};

    /// Linear thread-id
    int linear_tid;

    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage&                  temp_storage;       ///< Reference to temp_storage
    WrappedScanInputIteratorT      d_in;               ///< Input data
    WrappedTransformInputIteratorT d_trans_in;         ///< Input data
    OutputIteratorT                d_block_sum;        ///< Output data (block reduction)
    ScanOpT                        scan_op;            ///< Binary scan operator
    ReductionOpT                   reduction_op;       ///< Binary reduction operator
    TransformOpT                   transform_op;       ///< Transformation operator
    InitValueT                     init_value;         ///< The init_value element for ScanOpT


    //---------------------------------------------------------------------
    // Block scan utility methods
    //---------------------------------------------------------------------

    /**
     * Exclusive scan specialization (first tile)
     */
    __device__ __forceinline__
    void ScanTile(
        ScanInputT          (&items)[ITEMS_PER_THREAD],
        ScanInputT          init_value,
        ScanOpT             scan_op,
        ScanOutputT         &block_aggregate,
        Int2Type<false>     /*is_inclusive*/)
    {
        BlockScanT(temp_storage.scan_reduce_storage.scan).ExclusiveScan(items, items, init_value, scan_op, block_aggregate);
        block_aggregate = scan_op(init_value, block_aggregate);
    }


    /**
     * Inclusive scan specialization (first tile)
     */
    __device__ __forceinline__
    void ScanTile(
        ScanInputT          (&items)[ITEMS_PER_THREAD],
        InitValueT          /*init_value*/,
        ScanOpT             scan_op,
        ScanInputT          &block_aggregate,
        Int2Type<true>      /*is_inclusive*/)
    {
        BlockScanT(temp_storage.scan_reduce_storage.scan).InclusiveScan(items, items, scan_op, block_aggregate);
    }


    /**
     * Exclusive scan specialization (subsequent tiles)
     */
    template <typename PrefixCallback>
    __device__ __forceinline__
    void ScanTile(
        ScanInputT          (&items)[ITEMS_PER_THREAD],
        ScanOpT             scan_op,
        PrefixCallback      &prefix_op,
        Int2Type<false>     /*is_inclusive*/)
    {
        BlockScanT(temp_storage.scan_reduce_storage.scan).ExclusiveScan(items, items, scan_op, prefix_op);
    }


    /**
     * Inclusive scan specialization (subsequent tiles)
     */
    template <typename PrefixCallback>
    __device__ __forceinline__
    void ScanTile(
        ScanInputT          (&items)[ITEMS_PER_THREAD],
        ScanOpT             scan_op,
        PrefixCallback      &prefix_op,
        Int2Type<true>      /*is_inclusive*/)
    {
        BlockScanT(temp_storage.scan_reduce_storage.scan).InclusiveScan(items, items, scan_op, prefix_op);
    }


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    AgentScanReduce(
        TempStorage&            temp_storage, ///< Reference to temp_storage
        ScanInputIteratorT      d_in,         ///< Input data
        TransformInputIteratorT d_trans_in,   ///< Input data
        OutputIteratorT         d_block_sum,  ///< Output data (block reduction)
        ScanOpT                 scan_op,      ///< Binary scan operator
        ReductionOpT            reduction_op, ///< Binary reduction operator
        TransformOpT            transform_op, ///< Transformation operator
        InitValueT              init_value)   ///< Initial value to seed the exclusive scan
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_trans_in(d_trans_in),
        d_block_sum(d_block_sum),
        scan_op(scan_op),
        reduction_op(reduction_op),
        transform_op(transform_op),
        init_value(init_value),
        linear_tid(RowMajorTid(AgentScanReducePolicyT::BLOCK_THREADS, 1, 1))
    {}


    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------

    /**
     * Process a tile of input (dynamic chained scan)
     */
    template <bool IS_LAST_TILE>                ///< Whether the current tile is the last tile
    __device__ __forceinline__ void ConsumeTile(
        OffsetT             num_remaining,      ///< Number of global input items remaining (including this tile)
        int                 tile_idx,           ///< Tile index
        OffsetT             tile_offset,        ///< Tile offset
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        // Load items
        ScanInputT items[ITEMS_PER_THREAD];
        TransformInputT trans_items[ITEMS_PER_THREAD];

        if (IS_LAST_TILE)
        {
            // Fill last element with the first element because collectives are
            // not suffix guarded.
            BlockLoadScanT(temp_storage.load)
              .Load(d_in + tile_offset,
                    items,
                    num_remaining,
                    *(d_in + tile_offset));
        }
        else
        {
            BlockLoadScanT(temp_storage.load).Load(d_in + tile_offset, items);
        }

        CTA_SYNC();

        // Perform tile scan
        if (tile_idx == 0)
        {
            // Scan first tile
            ScanOutputT block_aggregate;
            ScanTile(items, init_value, scan_op, block_aggregate, Int2Type<IS_INCLUSIVE>());
            if ((!IS_LAST_TILE) && (threadIdx.x == 0))
                tile_state.SetInclusive(0, block_aggregate);
        }
        else
        {
            // Scan non-first tile
            TilePrefixCallbackOpT prefix_op(tile_state, temp_storage.scan_reduce_storage.prefix, scan_op, tile_idx);
            ScanTile(items, scan_op, prefix_op, Int2Type<IS_INCLUSIVE>());
        }

        CTA_SYNC();

        // Perform reduce and store
        ReduceOutputT item = ReduceOutputT();
        ReduceOutputT block_aggregate = ReduceOutputT();
        ReduceOutputT thread_aggregate = ReduceOutputT();

        if (IS_LAST_TILE) { // partially-full tile

            BlockStoreScanT(temp_storage.store).Exchange(tile_offset, items, num_remaining);

            // load additional data
            BlockLoadTransformT(temp_storage.trans_load)
              .Load(d_trans_in + tile_offset,
                    trans_items,
                    num_remaining,
                    *(d_trans_in + tile_offset));

            // transpose data in a partially-full tile
            //BlockStoreScanT(temp_storage.store).Exchange(tile_offset, items, num_remaining);
            BlockStoreTransformT(temp_storage.trans_store).Exchange(tile_offset, trans_items, num_remaining);

            CTA_SYNC();

	    // thread transform-reduction (TODO generalize)
            int tid         = linear_tid & (CUB_PTX_WARP_THREADS - 1);
            int wid         = linear_tid >> CUB_PTX_LOG_WARP_THREADS;
            int warp_offset = wid * CUB_PTX_WARP_THREADS * ITEMS_PER_THREAD;

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                if ((ITEM * CUB_PTX_WARP_THREADS) + warp_offset + tid < num_remaining)
                {
                    item = transform_op(items[ITEM], trans_items[ITEM]);
                    thread_aggregate = reduction_op(thread_aggregate, item);
                }
            }

        } else { // full tile

            // load additional data
            BlockLoadTransformT(temp_storage.trans_load).Load(d_trans_in + tile_offset, trans_items);

            CTA_SYNC();

            // thread transform-reduction
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                item = transform_op(items[ITEM], trans_items[ITEM]);
                thread_aggregate = reduction_op(thread_aggregate, item);
            }

        }

        // block reduction
        block_aggregate = BlockReduceT(temp_storage.scan_reduce_storage.reduce).Reduce(thread_aggregate, reduction_op);

        if (threadIdx.x == 0)
            d_block_sum[blockIdx.x] = block_aggregate;

    }


    /**
     * Scan tiles of items as part of a dynamic chained scan
     */
    __device__ __forceinline__ void ConsumeRange(
        OffsetT             num_items,          ///< Total number of input items
        ScanTileStateT&     tile_state,         ///< Global tile state descriptor
        int                 start_tile)         ///< The starting tile for the current grid
    {
        // Blocks are launched in increasing order, so just assign one tile per block
        int     tile_idx        = start_tile + blockIdx.x;          // Current tile index
        OffsetT tile_offset     = OffsetT(TILE_ITEMS) * tile_idx;   // Global offset for the current tile
        OffsetT num_remaining   = num_items - tile_offset;          // Remaining items (including this tile)

        if (num_remaining > TILE_ITEMS)
        {
            // Not last tile
            ConsumeTile<false>(num_remaining, tile_idx, tile_offset, tile_state);
        }
        else if (num_remaining > 0)
        {
            // Last tile
            ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state);
        }
    }

};


CUB_NAMESPACE_END

