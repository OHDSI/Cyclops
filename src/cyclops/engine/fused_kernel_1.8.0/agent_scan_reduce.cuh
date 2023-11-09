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
#include "block_store_exchange.cuh"
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_namespace.cuh>

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentScan
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    BlockStoreAlgorithm         _STORE_ALGORITHM,               ///< The BlockStore algorithm to use
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
    //BlockReduceAlgorithm        _REDUCE_ALGORITHM>               ///< Cooperative block-wide reduction algorithm to use
struct AgentScanPolicy1
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
//	VECTOR_LOAD_LENGTH      = _VECTOR_LOAD_LENGTH,          ///< Number of items per vectorized load
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;          ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;           ///< Cache load modifier for reading input elements
    static const BlockStoreAlgorithm    STORE_ALGORITHM         = _STORE_ALGORITHM;         ///< The BlockStore algorithm to use
    static const BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;          ///< The BlockScan algorithm to use
    //static const BlockReduceAlgorithm   REDUCE_ALGORITHM        = _REDUCE_ALGORITHM;        ///< Cooperative block-wide reduction algorithm to use
};




/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief AgentScan implements a stateful abstraction of CUDA thread blocks for participating in device-wide prefix scan .
 */
template <
    typename AgentScanPolicyT,      ///< Parameterized AgentScanPolicyT tuning policy type
    typename ScanInputIteratorT,        ///< Random-access input iterator type
    typename TransformInputIteratorT,       ///< Random-access additional input iterator type
    typename OutputIteratorT,       ///< Random-access reduction output iterator type
    typename ScanOpT,               ///< Scan functor type
    typename ReductionOpT,          ///< Reduction functor type
    typename TransformOpT,          ///< Transformation functor type
    typename InitValueT,            ///< The init_value element for ScanOpT type (cub::NullType for inclusive scan)
    typename OffsetT>               ///< Signed integer type for global offsets
struct AgentScanReduce
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // The input value type
    typedef typename std::iterator_traits<ScanInputIteratorT>::value_type InputT;
    typedef typename std::iterator_traits<TransformInputIteratorT>::value_type InputT1;
   
    // The output value type
    typedef typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<ScanInputIteratorT>::value_type,                                          // ... then the input iterator's value type,
        typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type

    // Tile status descriptor interface type
    typedef ScanTileState<InputT> ScanTileStateT;

    // Input iterator wrapper type (for applying cache modifier)
    typedef typename If<IsPointer<ScanInputIteratorT>::VALUE,
            CacheModifiedInputIterator<AgentScanPolicyT::LOAD_MODIFIER, InputT, OffsetT>,   // Wrap the native input pointer with CacheModifiedInputIterator
            ScanInputIteratorT>::Type                                                           // Directly use the supplied input iterator type
        WrappedScanInputIteratorT;

    typedef typename If<IsPointer<TransformInputIteratorT>::VALUE,
            CacheModifiedInputIterator<AgentScanPolicyT::LOAD_MODIFIER, InputT1, OffsetT>,   // Wrap the native input pointer with CacheModifiedInputIterator
            ScanInputIteratorT>::Type                                                           // Directly use the supplied input iterator type
        WrappedTransformInputIteratorT;

    // Constants
    enum
    {
        IS_INCLUSIVE        = Equals<InitValueT, NullType>::VALUE,            // Inclusive scan if no init_value type is provided
        BLOCK_THREADS       = AgentScanPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD    = AgentScanPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Parameterized BlockLoad type
    typedef BlockLoad<
            InputT,
            AgentScanPolicyT::BLOCK_THREADS,
            AgentScanPolicyT::ITEMS_PER_THREAD,
            AgentScanPolicyT::LOAD_ALGORITHM>
        BlockLoadT;

    typedef BlockLoad<
            InputT1,
            AgentScanPolicyT::BLOCK_THREADS,
            AgentScanPolicyT::ITEMS_PER_THREAD,
            AgentScanPolicyT::LOAD_ALGORITHM>
        BlockLoadT1;

    // Parameterized BlockStore type
    typedef BlockStore<
            InputT,
            AgentScanPolicyT::BLOCK_THREADS,
            AgentScanPolicyT::ITEMS_PER_THREAD,
            AgentScanPolicyT::STORE_ALGORITHM>
        BlockStoreT;
    typedef BlockStore<
            InputT1,
            AgentScanPolicyT::BLOCK_THREADS,
            AgentScanPolicyT::ITEMS_PER_THREAD,
            AgentScanPolicyT::STORE_ALGORITHM>
        BlockStoreT1;

    // Parameterized BlockReduce type
    typedef BlockReduce<
            OutputT,
            AgentScanPolicyT::BLOCK_THREADS,
            BLOCK_REDUCE_WARP_REDUCTIONS>
        BlockReduceT;

    // Parameterized BlockScan type
    typedef BlockScan<
            InputT,
            AgentScanPolicyT::BLOCK_THREADS,
            AgentScanPolicyT::SCAN_ALGORITHM>
        BlockScanT;

    // Callback type for obtaining tile prefix during block scan
    typedef TilePrefixCallbackOp<
            InputT,
            ScanOpT,
            ScanTileStateT>
        TilePrefixCallbackOpT;

    // Stateful BlockScan prefix callback type for managing a running total while scanning consecutive tiles
    typedef BlockScanRunningPrefixOp<
            InputT,
            ScanOpT>
        RunningPrefixCallbackOp;

    // Shared memory type for this thread block
    union _TempStorage
    {
        typename BlockLoadT::TempStorage    load;       // Smem needed for tile loading
        typename BlockStoreT::TempStorage   store;      // Smem needed for tile storing
        typename BlockLoadT1::TempStorage    load1;       // Smem needed for tile loading
        typename BlockStoreT1::TempStorage   store1;      // Smem needed for tile storing

        struct
        {
            typename TilePrefixCallbackOpT::TempStorage  prefix;     // Smem needed for cooperative prefix callback
            typename BlockScanT::TempStorage             scan;       // Smem needed for tile scanning
            typename BlockReduceT::TempStorage           reduce;
        };
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};

    /// Linear thread-id
    int linear_tid;

    
    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage&               temp_storage;       ///< Reference to temp_storage
    WrappedScanInputIteratorT       d_in;               ///< Input data
    WrappedTransformInputIteratorT      d_trans_in;              ///< Additional input data for transformation
    OutputIteratorT             d_block_sum;        ///< Output data (block reduction)
    OutputIteratorT             d_sum;              ///< Output data (reduction)
    ScanOpT                     scan_op;            ///< Binary scan operator
    ReductionOpT                reduction_op;       ///< Binary reduction operator
    TransformOpT                transform_op;       ///< Transformation operator on scan output    
    InitValueT                  init_value;         ///< The init_value element for ScanOpT

   
    //---------------------------------------------------------------------
    // Block scan utility methods
    //---------------------------------------------------------------------

    /**
     * Exclusive scan specialization (first tile)
     */
    __device__ __forceinline__
    void ScanTile(
        OutputT             (&items)[ITEMS_PER_THREAD],
        OutputT             init_value,
        ScanOpT             scan_op,
        OutputT             &block_aggregate,
        Int2Type<false>     /*is_inclusive*/)
    {
        BlockScanT(temp_storage.scan).ExclusiveScan(items, items, init_value, scan_op, block_aggregate);
        block_aggregate = scan_op(init_value, block_aggregate);
    }


    /**
     * Inclusive scan specialization (first tile)
     */
    __device__ __forceinline__
    void ScanTile(
        InputT             (&items)[ITEMS_PER_THREAD],
        InitValueT          /*init_value*/,
        ScanOpT             scan_op,
        InputT             &block_aggregate,
        Int2Type<true>      /*is_inclusive*/)
    {
        BlockScanT(temp_storage.scan).InclusiveScan(items, items, scan_op, block_aggregate);
    }


    /**
     * Exclusive scan specialization (subsequent tiles)
     */
    template <typename PrefixCallback>
    __device__ __forceinline__
    void ScanTile(
        OutputT             (&items)[ITEMS_PER_THREAD],
        ScanOpT             scan_op,
        PrefixCallback      &prefix_op,
        Int2Type<false>     /*is_inclusive*/)
    {
        BlockScanT(temp_storage.scan).ExclusiveScan(items, items, scan_op, prefix_op);
    }


    /**
     * Inclusive scan specialization (subsequent tiles)
     */
    template <typename PrefixCallback>
    __device__ __forceinline__
    void ScanTile(
        InputT             (&items)[ITEMS_PER_THREAD],
        ScanOpT             scan_op,
        PrefixCallback      &prefix_op,
        Int2Type<true>      /*is_inclusive*/)
    {
        BlockScanT(temp_storage.scan).InclusiveScan(items, items, scan_op, prefix_op);
    }


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    AgentScanReduce(
        TempStorage&    temp_storage,       ///< Reference to temp_storage
        ScanInputIteratorT  d_in,               ///< Input data
        TransformInputIteratorT d_trans_in,              ///< Additional input data for transformation
        OutputIteratorT d_block_sum,        ///< Output data (block reduction)
        OutputIteratorT d_sum,              ///< Output data (reduction)
        ScanOpT         scan_op,            ///< Binary scan operator
        ReductionOpT    reduction_op,       ///< Binary reduction operator
        TransformOpT    transform_op,       ///< Transformation operator on scan output
        InitValueT      init_value)         ///< Initial value to seed the exclusive scan
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_trans_in(d_trans_in),
        d_block_sum(d_block_sum),
        d_sum(d_sum),
        scan_op(scan_op),
        reduction_op(reduction_op),
        transform_op(transform_op),
        init_value(init_value),
        linear_tid(RowMajorTid(AgentScanPolicyT::BLOCK_THREADS, 1, 1))
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
        InputT items[ITEMS_PER_THREAD]; // input for scan
        InputT1 items1[ITEMS_PER_THREAD]; // input for reduction

        if (IS_LAST_TILE)
            BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items, num_remaining);
        else
            BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items);

        CTA_SYNC();

        // Perform tile scan
        if (tile_idx == 0)
        {
            // Scan first tile
            InputT block_aggregate;
            ScanTile(items, init_value, scan_op, block_aggregate, Int2Type<IS_INCLUSIVE>());
            if ((!IS_LAST_TILE) && (threadIdx.x == 0))
                tile_state.SetInclusive(0, block_aggregate);
        }
        else
        {
            // Scan non-first tile
            TilePrefixCallbackOpT prefix_op(tile_state, temp_storage.prefix, scan_op, tile_idx);
            ScanTile(items, scan_op, prefix_op, Int2Type<IS_INCLUSIVE>());
        }

        CTA_SYNC();

        // Perform reduce and store
        OutputT item;
        OutputT block_aggregate = OutputT();
        OutputT thread_aggregate = OutputT();

        if (IS_LAST_TILE) { // partially-full tile

            BlockStoreT(temp_storage.store).Exchange(tile_offset, items, num_remaining);
            
            // load additional data
            BlockLoadT1(temp_storage.load1).Load(d_trans_in + tile_offset, items1, num_remaining);
            BlockStoreT1(temp_storage.store1).Exchange(tile_offset, items1, num_remaining);
	    
            CTA_SYNC();

            // thread transform reduction
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                if ((ITEM * BLOCK_THREADS) + linear_tid < num_remaining)
                {
                    item = transform_op(items[ITEM], items1[ITEM]);
                    thread_aggregate = reduction_op(thread_aggregate, item);
                }
            }

        } else { // full tile

            // load additional data
            BlockLoadT1(temp_storage.load1).Load(d_trans_in + tile_offset, items1);
            
            CTA_SYNC();

            // thread transform reduction
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                item = transform_op(items[ITEM], items1[ITEM]);
                thread_aggregate = reduction_op(thread_aggregate, item);
            }

        }

        // block reduction
        block_aggregate = BlockReduceT(temp_storage.reduce).Reduce(thread_aggregate, reduction_op);

        if (threadIdx.x == 0)
            d_block_sum[blockIdx.x] = block_aggregate;

    }


    /**
     * Scan tiles of items as part of a dynamic chained scan
     */
    __device__ __forceinline__ void ConsumeRange(
        int                 num_items,          ///< Total number of input items
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

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

