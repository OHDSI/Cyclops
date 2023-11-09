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
 * Operations for writing linear segments of data from the CUDA thread block
 */

#pragma once

#include <iterator>
#include <type_traits>

#include <cub/block/block_exchange.cuh>
#include <cub/block/block_store.cuh>
#include <cub/config.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

/**
 * \addtogroup UtilIo
 * @{
 */

/**
 * \brief The BlockStore class provides [<em>collective</em>](index.html#sec0) data movement methods for writing a [<em>blocked arrangement</em>](index.html#sec5sec3) of items partitioned across a CUDA thread block to a linear segment of memory.  ![](block_store_logo.png)
 * \ingroup BlockModule
 * \ingroup UtilIo
 *
 * \tparam T                    The type of data to be written.
 * \tparam BLOCK_DIM_X          The thread block length in threads along the X dimension
 * \tparam ITEMS_PER_THREAD     The number of consecutive items partitioned onto each thread.
 * \tparam ALGORITHM            <b>[optional]</b> cub::BlockStoreAlgorithm tuning policy enumeration.  default: cub::BLOCK_STORE_DIRECT.
 * \tparam BLOCK_DIM_Y          <b>[optional]</b> The thread block length in threads along the Y dimension (default: 1)
 * \tparam BLOCK_DIM_Z          <b>[optional]</b> The thread block length in threads along the Z dimension (default: 1)
 * \tparam PTX_ARCH             <b>[optional]</b> \ptxversion
 *
 * \par Overview
 * - The BlockStore class provides a single data movement abstraction that can be specialized
 *   to implement different cub::BlockStoreAlgorithm strategies.  This facilitates different
 *   performance policies for different architectures, data types, granularity sizes, etc.
 * - BlockStore can be optionally specialized by different data movement strategies:
 *   -# <b>cub::BLOCK_STORE_DIRECT</b>.  A [<em>blocked arrangement</em>](index.html#sec5sec3) of data is written
 *      directly to memory. [More...](\ref cub::BlockStoreAlgorithm)
 *   -# <b>cub::BLOCK_STORE_STRIPED</b>.  A [<em>striped arrangement</em>](index.html#sec5sec3)
 *      of data is written directly to memory. [More...](\ref cub::BlockStoreAlgorithm)
 *   -# <b>cub::BLOCK_STORE_VECTORIZE</b>.  A [<em>blocked arrangement</em>](index.html#sec5sec3)
 *      of data is written directly to memory using CUDA's built-in vectorized stores as a
 *      coalescing optimization.  [More...](\ref cub::BlockStoreAlgorithm)
 *   -# <b>cub::BLOCK_STORE_TRANSPOSE</b>.  A [<em>blocked arrangement</em>](index.html#sec5sec3)
 *      is locally transposed into a [<em>striped arrangement</em>](index.html#sec5sec3) which is
 *      then written to memory.  [More...](\ref cub::BlockStoreAlgorithm)
 *   -# <b>cub::BLOCK_STORE_WARP_TRANSPOSE</b>.  A [<em>blocked arrangement</em>](index.html#sec5sec3)
 *      is locally transposed into a [<em>warp-striped arrangement</em>](index.html#sec5sec3) which is
 *      then written to memory.  [More...](\ref cub::BlockStoreAlgorithm)
 *   -# <b>cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED</b>.  A [<em>blocked arrangement</em>](index.html#sec5sec3)
 *      is locally transposed into a [<em>warp-striped arrangement</em>](index.html#sec5sec3) which is
 *      then written to memory. To reduce the shared memory requireent, only one warp's worth of shared
 *      memory is provisioned and is subsequently time-sliced among warps.  [More...](\ref cub::BlockStoreAlgorithm)
 * - \rowmajor
 *
 * \par A Simple Example
 * \blockcollective{BlockStore}
 * \par
 * The code snippet below illustrates the storing of a "blocked" arrangement
 * of 512 integers across 128 threads (where each thread owns 4 consecutive items)
 * into a linear segment of memory.  The store is specialized for \p BLOCK_STORE_WARP_TRANSPOSE,
 * meaning items are locally reordered among threads so that memory references will be
 * efficiently coalesced using a warp-striped access pattern.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/block/block_store.cuh>
 *
 * __global__ void ExampleKernel(int *d_data, ...)
 * {
 *     // Specialize BlockStore for a 1D block of 128 threads owning 4 integer items each
 *     typedef cub::BlockStore<int, 128, 4, BLOCK_STORE_WARP_TRANSPOSE> BlockStore;
 *
 *     // Allocate shared memory for BlockStore
 *     __shared__ typename BlockStore::TempStorage temp_storage;
 *
 *     // Obtain a segment of consecutive items that are blocked across threads
 *     int thread_data[4];
 *     ...
 *
 *     // Store items to linear memory
 *     BlockStore(temp_storage).Store(d_data, thread_data);
 *
 * \endcode
 * \par
 * Suppose the set of \p thread_data across the block of threads is
 * <tt>{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }</tt>.
 * The output \p d_data will be <tt>0, 1, 2, 3, 4, 5, ...</tt>.
 *
 * \par Re-using dynamically allocating shared memory
 * The following example under the examples/block folder illustrates usage of
 * dynamically shared memory with BlockReduce and how to re-purpose
 * the same memory region:
 * <a href="../../examples/block/example_block_reduce_dyn_smem.cu">example_block_reduce_dyn_smem.cu</a>
 *
 * This example can be easily adapted to the storage required by BlockStore.
 */
template <
    typename                T,
    int                     BLOCK_DIM_X,
    int                     ITEMS_PER_THREAD,
    BlockStoreAlgorithm     ALGORITHM           = BLOCK_STORE_DIRECT,
    int                     BLOCK_DIM_Y         = 1,
    int                     BLOCK_DIM_Z         = 1,
    int                     PTX_ARCH            = CUB_PTX_ARCH>
class BlockStoreExchange
{
private:
    /******************************************************************************
     * Constants and typed definitions
     ******************************************************************************/

    /// Constants
    enum
    {
        /// The thread block size in threads
        BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
    };


    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /// Store helper
    template <BlockStoreAlgorithm _POLICY, int DUMMY>
    struct StoreInternal;


    /**
     * BLOCK_STORE_DIRECT specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_DIRECT, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType TempStorage;

        /// Linear thread-id
        int linear_tid;

        /// Constructor
        __device__ __forceinline__ StoreInternal(
            TempStorage &/*temp_storage*/,
            int linear_tid)
        :
            linear_tid(linear_tid)
        {}

        /// Store items into a linear segment of memory
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Store(
            OutputIteratorT     block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            StoreDirectBlocked(linear_tid, block_itr, items);
        }

        /// Store items into a linear segment of memory, guarded by range
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Store(
            OutputIteratorT     block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            int                 valid_items)                ///< [in] Number of valid items to write
        {
            StoreDirectBlocked(linear_tid, block_itr, items, valid_items);
        }
    };


    /**
    * BLOCK_STORE_STRIPED specialization of store helper
    */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_STRIPED, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType TempStorage;

        /// Linear thread-id
        int linear_tid;

        /// Constructor
        __device__ __forceinline__ StoreInternal(
            TempStorage &/*temp_storage*/,
            int linear_tid)
        :
            linear_tid(linear_tid)
        {}

        /// Store items into a linear segment of memory
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Store(
            OutputIteratorT     block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            StoreDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items);
        }

        /// Store items into a linear segment of memory, guarded by range
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Store(
            OutputIteratorT   block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            int                 valid_items)                ///< [in] Number of valid items to write
        {
            StoreDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items, valid_items);
        }
    };


    /**
     * BLOCK_STORE_VECTORIZE specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_VECTORIZE, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType TempStorage;

        /// Linear thread-id
        int linear_tid;

        /// Constructor
        __device__ __forceinline__ StoreInternal(
            TempStorage &/*temp_storage*/,
            int linear_tid)
        :
            linear_tid(linear_tid)
        {}

        /// Store items into a linear segment of memory, specialized for native pointer types (attempts vectorization)
        __device__ __forceinline__ void Store(
            T                   *block_ptr,                 ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            StoreDirectBlockedVectorized(linear_tid, block_ptr, items);
        }

        /// Store items into a linear segment of memory, specialized for opaque input iterators (skips vectorization)
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Store(
            OutputIteratorT    block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            StoreDirectBlocked(linear_tid, block_itr, items);
        }

        /// Store items into a linear segment of memory, guarded by range
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Store(
            OutputIteratorT     block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            int                 valid_items)                ///< [in] Number of valid items to write
        {
            StoreDirectBlocked(linear_tid, block_itr, items, valid_items);
        }
    };


    /**
     * BLOCK_STORE_TRANSPOSE specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_TRANSPOSE, DUMMY>
    {
        // BlockExchange utility type for keys
        typedef BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD, false, BLOCK_DIM_Y, BLOCK_DIM_Z, PTX_ARCH> BlockExchange;

        /// Shared memory storage layout type
        struct _TempStorage : BlockExchange::TempStorage
        {
            /// Temporary storage for partially-full block guard
            volatile int valid_items;
        };

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &temp_storage;

        /// Linear thread-id
        int linear_tid;

        /// Constructor
        __device__ __forceinline__ StoreInternal(
            TempStorage &temp_storage,
            int linear_tid)
        :
            temp_storage(temp_storage.Alias()),
            linear_tid(linear_tid)
        {}

        /// Store items into a linear segment of memory
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Store(
            OutputIteratorT     block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            BlockExchange(temp_storage).BlockedToStriped(items);
            StoreDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items);
        }

        /// Store items into a linear segment of memory, guarded by range
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Store(
            OutputIteratorT   block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            int                 valid_items)                ///< [in] Number of valid items to write
        {
            BlockExchange(temp_storage).BlockedToStriped(items);
            if (linear_tid == 0)
                temp_storage.valid_items = valid_items;     // Move through volatile smem as a workaround to prevent RF spilling on subsequent loads
            CTA_SYNC();
            StoreDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items, temp_storage.valid_items);
        }

        /// Exchange items for partially-full tile (last tile), guarded by range
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Exchange(
            OutputIteratorT   block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to exchange
            int                 valid_items)                ///< [in] Number of valid items to write
        {
            BlockExchange(temp_storage).BlockedToStriped(items);
        }
    };


    /**
     * BLOCK_STORE_WARP_TRANSPOSE specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_WARP_TRANSPOSE, DUMMY>
    {
        enum
        {
            WARP_THREADS = CUB_WARP_THREADS(PTX_ARCH)
        };

        // Assert BLOCK_THREADS must be a multiple of WARP_THREADS
        CUB_STATIC_ASSERT((int(BLOCK_THREADS) % int(WARP_THREADS) == 0), "BLOCK_THREADS must be a multiple of WARP_THREADS");

        // BlockExchange utility type for keys
        typedef BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD, false, BLOCK_DIM_Y, BLOCK_DIM_Z, PTX_ARCH> BlockExchange;

        /// Shared memory storage layout type
        struct _TempStorage : BlockExchange::TempStorage
        {
            /// Temporary storage for partially-full block guard
            volatile int valid_items;
        };

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &temp_storage;

        /// Linear thread-id
        int linear_tid;

        /// Constructor
        __device__ __forceinline__ StoreInternal(
            TempStorage &temp_storage,
            int linear_tid)
        :
            temp_storage(temp_storage.Alias()),
            linear_tid(linear_tid)
        {}

        /// Store items into a linear segment of memory
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Store(
            OutputIteratorT   block_itr,                    ///< [in] The thread block's base output iterator for storing to
            T                 (&items)[ITEMS_PER_THREAD])   ///< [in] Data to store
        {
            BlockExchange(temp_storage).BlockedToWarpStriped(items);
            StoreDirectWarpStriped(linear_tid, block_itr, items);
        }

        /// Store items into a linear segment of memory, guarded by range
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Store(
            OutputIteratorT   block_itr,                    ///< [in] The thread block's base output iterator for storing to
            T                 (&items)[ITEMS_PER_THREAD],   ///< [in] Data to store
            int               valid_items)                  ///< [in] Number of valid items to write
        {
            BlockExchange(temp_storage).BlockedToWarpStriped(items);
            if (linear_tid == 0)
                temp_storage.valid_items = valid_items;     // Move through volatile smem as a workaround to prevent RF spilling on subsequent loads
            CTA_SYNC();
            StoreDirectWarpStriped(linear_tid, block_itr, items, temp_storage.valid_items);
        }

        /// Exchange items for partially-full tile (last tile), guarded by range
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Exchange(
            OutputIteratorT   block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to exchange
            int                 valid_items)                ///< [in] Number of valid items to write
        {
            BlockExchange(temp_storage).BlockedToWarpStriped(items);
        }
    };


    /**
     * BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED, DUMMY>
    {
        enum
        {
            WARP_THREADS = CUB_WARP_THREADS(PTX_ARCH)
        };

        // Assert BLOCK_THREADS must be a multiple of WARP_THREADS
        CUB_STATIC_ASSERT((int(BLOCK_THREADS) % int(WARP_THREADS) == 0), "BLOCK_THREADS must be a multiple of WARP_THREADS");

        // BlockExchange utility type for keys
        typedef BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD, true, BLOCK_DIM_Y, BLOCK_DIM_Z, PTX_ARCH> BlockExchange;

        /// Shared memory storage layout type
        struct _TempStorage : BlockExchange::TempStorage
        {
            /// Temporary storage for partially-full block guard
            volatile int valid_items;
        };

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &temp_storage;

        /// Linear thread-id
        int linear_tid;

        /// Constructor
        __device__ __forceinline__ StoreInternal(
            TempStorage &temp_storage,
            int linear_tid)
        :
            temp_storage(temp_storage.Alias()),
            linear_tid(linear_tid)
        {}

        /// Store items into a linear segment of memory
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Store(
            OutputIteratorT     block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            BlockExchange(temp_storage).BlockedToWarpStriped(items);
            StoreDirectWarpStriped(linear_tid, block_itr, items);
        }

        /// Store items into a linear segment of memory, guarded by range
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Store(
            OutputIteratorT   block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            int                 valid_items)                ///< [in] Number of valid items to write
        {
            BlockExchange(temp_storage).BlockedToWarpStriped(items);
            if (linear_tid == 0)
                temp_storage.valid_items = valid_items;     // Move through volatile smem as a workaround to prevent RF spilling on subsequent loads
            CTA_SYNC();
            StoreDirectWarpStriped(linear_tid, block_itr, items, temp_storage.valid_items);
        }

        /// Exchange items for partially-full tile (last tile), guarded by range
        template <typename OutputIteratorT>
        __device__ __forceinline__ void Exchange(
            OutputIteratorT   block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to exchange
            int                 valid_items)                ///< [in] Number of valid items to write
        {
            BlockExchange(temp_storage).BlockedToWarpStriped(items);
        }
    };
 

    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Internal load implementation to use
    typedef StoreInternal<ALGORITHM, 0> InternalStore;


    /// Shared memory storage layout type
    typedef typename InternalStore::TempStorage _TempStorage;


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Thread reference to shared storage
    _TempStorage &temp_storage;

    /// Linear thread-id
    int linear_tid;

public:


    /// \smemstorage{BlockStore}
    struct TempStorage : Uninitialized<_TempStorage> {};


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.
     */
    __device__ __forceinline__ BlockStoreExchange()
    :
        temp_storage(PrivateStorage()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    __device__ __forceinline__ BlockStoreExchange(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}


    //@}  end member group
    /******************************************************************//**
     * \name Data movement
     *********************************************************************/
    //@{


    /**
     * \brief Store items into a linear segment of memory.
     *
     * \par
     * - \blocked
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates the storing of a "blocked" arrangement
     * of 512 integers across 128 threads (where each thread owns 4 consecutive items)
     * into a linear segment of memory.  The store is specialized for \p BLOCK_STORE_WARP_TRANSPOSE,
     * meaning items are locally reordered among threads so that memory references will be
     * efficiently coalesced using a warp-striped access pattern.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_store.cuh>
     *
     * __global__ void ExampleKernel(int *d_data, ...)
     * {
     *     // Specialize BlockStore for a 1D block of 128 threads owning 4 integer items each
     *     typedef cub::BlockStore<int, 128, 4, BLOCK_STORE_WARP_TRANSPOSE> BlockStore;
     *
     *     // Allocate shared memory for BlockStore
     *     __shared__ typename BlockStore::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Store items to linear memory
     *     int thread_data[4];
     *     BlockStore(temp_storage).Store(d_data, thread_data);
     *
     * \endcode
     * \par
     * Suppose the set of \p thread_data across the block of threads is
     * <tt>{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }</tt>.
     * The output \p d_data will be <tt>0, 1, 2, 3, 4, 5, ...</tt>.
     *
     */
    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(
        OutputIteratorT     block_itr,                  ///< [out] The thread block's base output iterator for storing to
        T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
    {
        InternalStore(temp_storage, linear_tid).Store(block_itr, items);
    }

    /**
     * \brief Store items into a linear segment of memory, guarded by range.
     *
     * \par
     * - \blocked
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates the guarded storing of a "blocked" arrangement
     * of 512 integers across 128 threads (where each thread owns 4 consecutive items)
     * into a linear segment of memory.  The store is specialized for \p BLOCK_STORE_WARP_TRANSPOSE,
     * meaning items are locally reordered among threads so that memory references will be
     * efficiently coalesced using a warp-striped access pattern.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_store.cuh>
     *
     * __global__ void ExampleKernel(int *d_data, int valid_items, ...)
     * {
     *     // Specialize BlockStore for a 1D block of 128 threads owning 4 integer items each
     *     typedef cub::BlockStore<int, 128, 4, BLOCK_STORE_WARP_TRANSPOSE> BlockStore;
     *
     *     // Allocate shared memory for BlockStore
     *     __shared__ typename BlockStore::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Store items to linear memory
     *     int thread_data[4];
     *     BlockStore(temp_storage).Store(d_data, thread_data, valid_items);
     *
     * \endcode
     * \par
     * Suppose the set of \p thread_data across the block of threads is
     * <tt>{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }</tt> and \p valid_items is \p 5.
     * The output \p d_data will be <tt>0, 1, 2, 3, 4, ?, ?, ?, ...</tt>, with
     * only the first two threads being unmasked to store portions of valid data.
     *
     */
    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(
        OutputIteratorT     block_itr,                  ///< [out] The thread block's base output iterator for storing to
        T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
        int                 valid_items)                ///< [in] Number of valid items to write
    {
        InternalStore(temp_storage, linear_tid).Store(block_itr, items, valid_items);
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Exchange(
        OutputIteratorT     block_itr,                  ///< [in] The thread block's base output iterator for storing to
        T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to exchange
        int                 valid_items)                ///< [in] Number of valid items to write
    {
        InternalStore(temp_storage, linear_tid).Exchange(block_itr, items, valid_items);
    }
    //@}  end member group
};

template <class Policy,
          class It,
          class T = typename std::iterator_traits<It>::value_type>
struct BlockStoreTypeExchange
{
  using type = cub::BlockStoreExchange<T,
                               Policy::BLOCK_THREADS,
                               Policy::ITEMS_PER_THREAD,
                               Policy::STORE_ALGORITHM>;
};

CUB_NAMESPACE_END

