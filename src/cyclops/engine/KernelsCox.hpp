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
    GpuModelSpecificsCox<BaseModel, RealType>::writecodeForScanLev1Kernel() {

        std::string name = "scan_lev1";

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel\n" <<
//             "__attribute__((reqd_work_group_size(psc_WG_SIZE, 1, 1)))\n" <<
             "void scan_lev1(\n" <<
             "    __global float *input_ary, __global float *output_ary,\n" <<
             "    __global float *restrict psc_partial_scan_buffer,\n" <<
             "    const int N,\n" <<
             "    const int psc_interval_size\n" <<
             "        , __global float *restrict psc_interval_results\n" <<
             "    ) {\n" <<
             "    // index psc_K in first dimension used for psc_carry storage\n" <<
             "        struct psc_wrapped_scan_type\n" <<
             "        {\n" <<
             "            float psc_value;\n" <<
             "        };\n" <<
//             "    // padded in psc_WG_SIZE to avoid bank conflicts\n" <<
             "    __local struct psc_wrapped_scan_type psc_ldata[psc_K + 1][psc_WG_SIZE + 1];\n" <<
             "    const int psc_interval_begin = psc_interval_size * get_group_id(0);\n" <<
             "    const int psc_interval_end   = min(psc_interval_begin + psc_interval_size, N);\n" <<
             "    const int psc_unit_size  = psc_K * psc_WG_SIZE; \n" << // 8192 = 32 * 256
             "    int psc_unit_base = psc_interval_begin;\n" <<
             "\n" <<
             "\n" <<
             "            for(; psc_unit_base + psc_unit_size <= psc_interval_end; psc_unit_base += psc_unit_size)\n" <<
             "\n" <<
             "        {\n" <<
             "\n" <<
             "            // {{{ read a unit's worth of data from psc_global\n" <<
             "\n" <<
             "            for(int psc_k = 0; psc_k < psc_K; psc_k++)\n" <<
             "            {\n" <<
             "                const int psc_offset = psc_k*psc_WG_SIZE + get_local_id(0);\n" <<
             "                const int psc_read_i = psc_unit_base + psc_offset;\n" <<
             "\n" <<
             "                {\n" <<
             "\n" <<
             "                    float psc_scan_value = (input_ary[psc_read_i]);\n" <<
             "\n" <<
             "                    const int psc_o_mod_k = psc_offset % psc_K;\n" <<
             "                    const int psc_o_div_k = psc_offset / psc_K;\n" <<
             "                    psc_ldata[psc_o_mod_k][psc_o_div_k].psc_value = psc_scan_value;\n" <<
             "\n" <<
             "                }\n" <<
             "            }\n" <<
             "\n" <<
//             "            pycl_printf(("after read from psc_global\n"));\n" <<
            "\n" <<
            "            // }}}\n" <<
            "\n" <<
            "            // {{{ psc_carry in from previous unit, if applicable\n" <<
            "\n" <<
            "\n" <<
            "            if (get_local_id(0) == 0 && psc_unit_base != psc_interval_begin)\n" <<
            "            {\n" <<
            "                float psc_tmp = psc_ldata[psc_K][psc_WG_SIZE - 1].psc_value;\n" <<
            "                float psc_tmp_aux = psc_ldata[0][0].psc_value;\n" <<
            "\n" <<
            "                psc_ldata[0][0].psc_value = psc_tmp + psc_tmp_aux;\n" <<
            "            }\n" <<
            "\n" <<
//            "            pycl_printf(("after psc_carry-in\n"));\n" <<
            "\n" <<
            "            // }}}\n" <<
            "\n" <<
            "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "\n" <<
            "            // {{{ scan along psc_k (sequentially in each work item)\n" <<
            "\n" <<
            "            float psc_sum = psc_ldata[0][get_local_id(0)].psc_value;\n" <<
            "\n" <<
            "\n" <<
            "            for (int psc_k = 1; psc_k < psc_K; psc_k++)\n" <<
            "            {\n" <<
            "                {\n" <<
            "                    float psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;\n" <<
            "\n" <<
            "\n" <<
            "                    psc_sum = psc_sum + psc_tmp;\n" <<
            "\n" <<
            "                    psc_ldata[psc_k][get_local_id(0)].psc_value = psc_sum;\n" <<
            "                }\n" <<
            "            }\n" <<
            "\n" <<
//            "            pycl_prfloatf(("after scan along psc_k\n"));\n" <<
            "\n" <<
            "            // }}}\n" <<
            "\n" <<
            "            // store psc_carry in out-of-bounds (padding) array entry (index psc_K) in\n" <<
            "            // the psc_K direction\n" <<
            "            psc_ldata[psc_K][get_local_id(0)].psc_value = psc_sum;\n" <<
            "\n" <<
            "\n" <<
            "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "\n" <<
            "            // {{{ tree-based local parallel scan\n" <<
            "\n" <<
            "            // This tree-based scan works as follows:\n" <<
            "            // - Each work item adds the previous item to its current state\n" <<
            "            // - barrier\n" <<
            "            // - Each work item adds in the item from two positions to the left\n" <<
            "            // - barrier\n" <<
            "            // - Each work item adds in the item from four positions to the left\n" <<
            "            // ...\n" <<
            "            // At the end, each item has summed all prior items.\n" <<
            "\n" <<
            "            // across psc_k groups, along local id\n" <<
            "            // (uses out-of-bounds psc_k=psc_K array entry for storage)\n" <<
            "\n" <<
            "            float psc_val = psc_ldata[psc_K][get_local_id(0)].psc_value;\n" <<
            "\n" <<
            "\n" <<
            "            for (int depth = 1; depth <= psc_WG_SIZE; depth<<=1) {\n" <<
            "\n" <<
            "                // {{{ reads from local allowed, writes to local not allowed\n" <<
            "\n" <<
            "                if (get_local_id(0) >= depth) {\n" <<
            "\n" <<
            "                    float psc_tmp = psc_ldata[psc_K][get_local_id(0) - depth].psc_value;\n" <<
            "\n" <<
            "                    {\n" <<
            "                        psc_val = psc_tmp + psc_val;\n" <<
            "                    }\n" <<
            "                }\n" <<
            "\n" <<
            "                // }}}\n" <<
            "\n" <<
            "                barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "\n" <<
            "                // {{{ writes to local allowed, reads from local not allowed\n" <<
            "\n" <<
            "                psc_ldata[psc_K][get_local_id(0)].psc_value = psc_val;\n" <<
            "\n" <<
            "                // }}}\n" <<
            "\n" <<
            "                barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "\n" <<
            "            }\n" <<
            "\n" <<
//            "            pycl_printf(("after tree scan\n"));\n" <<
            "\n" <<
            "            // }}}\n" <<
            "\n" <<
            "            // {{{ update local values\n" <<
            "\n" <<
            "            if (get_local_id(0) > 0)\n" <<
            "            {\n" <<
            "                psc_sum = psc_ldata[psc_K][get_local_id(0) - 1].psc_value;\n" <<
            "\n" <<
            "                for(int psc_k = 0; psc_k < psc_K; psc_k++)\n" <<
            "                {\n" <<
            "                    {\n" <<
            "                        float psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;\n" <<
            "                        psc_ldata[psc_k][get_local_id(0)].psc_value = psc_sum + psc_tmp;\n" <<
            "                    }\n" <<
            "                }\n" <<
            "            }\n" <<
            "\n" <<
            "\n" <<
//            "            pycl_printf(("after local update\n"));\n" <<
            "\n" <<
            "            // }}}\n" <<
            "\n" <<
            "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "\n" <<
            "            // {{{ write data\n" <<
            "\n" <<
            "            {\n" <<
            "                // work hard with index math to achieve contiguous 32-bit stores\n" <<
            "                __global int *psc_dest =\n" <<
            "                    (__global int *) (psc_partial_scan_buffer + psc_unit_base);\n" <<
            "\n" <<
            "\n" <<
            "\n" <<
            "                const int psc_scan_types_per_int = 1;\n" <<
            "\n" <<
            "\n" <<
            "                    for (int k = 0; k < psc_K; k++)\n" <<
            "                    {\n" <<
            "                        int psc_linear_index = k*psc_WG_SIZE + get_local_id(0);\n" <<
            "                        int psc_linear_scan_data_idx =\n" <<
            "                                psc_linear_index / psc_scan_types_per_int;\n" <<
            "                        int remainder =\n" <<
            "                                psc_linear_index - psc_linear_scan_data_idx * psc_scan_types_per_int;\n" <<
            "\n" <<
            "                        __local int *psc_src = (__local int *) &(\n" <<
            "                        psc_ldata\n" <<
            "                        [psc_linear_scan_data_idx % psc_K]\n" <<
            "                        [psc_linear_scan_data_idx / psc_K].psc_value);\n" <<
            "\n" <<
            "                        psc_dest[psc_linear_index] = psc_src[remainder];\n" <<
            "                    }\n" <<
            "\n" <<
            "\n" <<
            "            }\n" <<
            "\n" <<
//            "            pycl_printf(("after write\n"));\n" <<
            "\n" <<
            "            // }}}\n" <<
            "\n" <<
            "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "        }\n" <<
            "\n" <<
            "            if (psc_unit_base < psc_interval_end)\n" <<
            "\n" <<
            "        {\n" <<
            "\n" <<
            "            // {{{ read a unit's worth of data from psc_global\n" <<
            "\n" <<
            "            for(int psc_k = 0; psc_k < psc_K; psc_k++)\n" <<
            "            {\n" <<
            "                const int psc_offset = psc_k*psc_WG_SIZE + get_local_id(0);\n" <<
            "                const int psc_read_i = psc_unit_base + psc_offset;\n" <<
            "\n" <<
            "                if (psc_read_i < psc_interval_end)\n" <<
            "                {\n" <<
            "\n" <<
            "                    float psc_scan_value = (input_ary[psc_read_i]);\n" <<
            "\n" <<
            "                    const int psc_o_mod_k = psc_offset % psc_K;\n" <<
            "                    const int psc_o_div_k = psc_offset / psc_K;\n" <<
            "                    psc_ldata[psc_o_mod_k][psc_o_div_k].psc_value = psc_scan_value;\n" <<
            "\n" <<
            "                }\n" <<
            "            }\n" <<
            "\n" <<
//            "            pycl_printf(("after read from psc_global\n"));\n" <<
            "\n" <<
            "            // }}}\n" <<
            "\n" <<
            "            // {{{ psc_carry in from previous unit, if applicable\n" <<
            "\n" <<
            "\n" <<
            "            if (get_local_id(0) == 0 && psc_unit_base != psc_interval_begin)\n" <<
            "            {\n" <<
            "                float psc_tmp = psc_ldata[psc_K][psc_WG_SIZE - 1].psc_value;\n" <<
            "                float psc_tmp_aux = psc_ldata[0][0].psc_value;\n" <<
            "\n" <<
            "                psc_ldata[0][0].psc_value = psc_tmp + psc_tmp_aux;\n" <<
            "            }\n" <<
            "\n" <<
//            "            pycl_printf(("after psc_carry-in\n"));\n" <<
            "\n" <<
            "            // }}}\n" <<
            "\n" <<
            "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "\n" <<
            "            // {{{ scan along psc_k (sequentially in each work item)\n" <<
            "\n" <<
            "            float psc_sum = psc_ldata[0][get_local_id(0)].psc_value;\n" <<
            "\n" <<
            "                const int psc_offset_end = psc_interval_end - psc_unit_base;\n" <<
            "\n" <<
            "            for (int psc_k = 1; psc_k < psc_K; psc_k++)\n" <<
            "            {\n" <<
            "                if ((int) (psc_K * get_local_id(0) + psc_k) < psc_offset_end)\n" <<
            "                {\n" <<
            "                    float psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;\n" <<
            "\n" <<
            "\n" <<
            "                    psc_sum = psc_sum + psc_tmp;\n" <<
            "\n" <<
            "                    psc_ldata[psc_k][get_local_id(0)].psc_value = psc_sum;\n" <<
            "                }\n" <<
            "            }\n" <<
            "\n" <<
//            "            pycl_printf(("after scan along psc_k\n"));\n" <<
            "\n" <<
            "            // }}}\n" <<
            "\n" <<
            "            // store psc_carry in out-of-bounds (padding) array entry (index psc_K) in\n" <<
            "            // the psc_K direction\n" <<
            "            psc_ldata[psc_K][get_local_id(0)].psc_value = psc_sum;\n" <<
            "\n" <<
            "\n" <<
            "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "\n" <<
            "            // {{{ tree-based local parallel scan\n" <<
            "\n" <<
            "            // This tree-based scan works as follows:\n" <<
            "            // - Each work item adds the previous item to its current state\n" <<
            "            // - barrier\n" <<
            "            // - Each work item adds in the item from two positions to the left\n" <<
            "            // - barrier\n" <<
            "            // - Each work item adds in the item from four positions to the left\n" <<
            "            // ...\n" <<
            "            // At the end, each item has summed all prior items.\n" <<
            "\n" <<
            "            // across psc_k groups, along local id\n" <<
            "            // (uses out-of-bounds psc_k=psc_K array entry for storage)\n" <<
            "\n" <<
            "            float psc_val = psc_ldata[psc_K][get_local_id(0)].psc_value;\n" <<
            "\n" <<
            "\n" <<
            "                for (int depth = 1; depth <= psc_WG_SIZE; depth<<=1) {\n" <<
            "\n" <<
            "                    // {{{ reads from local allowed, writes to local not allowed\n" <<
            "\n" <<
            "                    if (get_local_id(0) >= depth)\n" <<
            "                    {\n" <<
            "                        float psc_tmp = psc_ldata[psc_K][get_local_id(0) - depth].psc_value;\n" <<
            "                        if (psc_K*get_local_id(0) < psc_offset_end)\n" <<
            "                        {\n" <<
            "                            psc_val = psc_tmp + psc_val;\n" <<
            "                        }\n" <<
            "\n" <<
            "                    }\n" <<
            "\n" <<
            "                    // }}}\n" <<
            "\n" <<
            "                    barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "\n" <<
            "                    // {{{ writes to local allowed, reads from local not allowed\n" <<
            "\n" <<
            "                    psc_ldata[psc_K][get_local_id(0)].psc_value = psc_val;\n" <<
            "\n" <<
            "                    // }}}\n" <<
            "\n" <<
            "                    barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "                }\n" <<
            "\n" <<
//            "            pycl_printf(("after tree scan\n"));\n" <<
            "\n" <<
            "            // }}}\n" <<
            "\n" <<
            "            // {{{ update local values\n" <<
            "\n" <<
            "            if (get_local_id(0) > 0)\n" <<
            "            {\n" <<
            "                psc_sum = psc_ldata[psc_K][get_local_id(0) - 1].psc_value;\n" <<
            "\n" <<
            "                for(int psc_k = 0; psc_k < psc_K; psc_k++)\n" <<
            "                {\n" <<
            "                    if (psc_K * get_local_id(0) + psc_k < psc_offset_end)\n" <<
            "                    {\n" <<
            "                        float psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;\n" <<
            "                        psc_ldata[psc_k][get_local_id(0)].psc_value = psc_sum + psc_tmp;\n" <<
            "                    }\n" <<
            "                }\n" <<
            "            }\n" <<
            "\n" <<
            "\n" <<
//            "            pycl_printf(("after local update\n"));\n" <<
            "\n" <<
            "            // }}}\n" <<
            "\n" <<
            "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "\n" <<
            "            // {{{ write data\n" <<
            "\n" <<
            "            {\n" <<
            "                // work hard with index math to achieve contiguous 32-bit stores\n" <<
            "                __global int *psc_dest =\n" <<
            "                    (__global int *) (psc_partial_scan_buffer + psc_unit_base);\n" <<
            "\n" <<
            "\n" <<
            "\n" <<
            "                const int psc_scan_types_per_int = 1;\n" <<
            "\n" <<
            "\n" <<
            "                for (int k = 0; k < psc_K; k++) {\n" <<
            "\n" <<
            "                    if (k*psc_WG_SIZE + get_local_id(0) < psc_scan_types_per_int*(psc_interval_end - psc_unit_base))\n" <<
            "                    {\n" <<
            "                        int psc_linear_index = k*psc_WG_SIZE + get_local_id(0);\n" <<
            "                        int psc_linear_scan_data_idx =\n" <<
            "                                psc_linear_index / psc_scan_types_per_int;\n" <<
            "                        int remainder =\n" <<
            "                                psc_linear_index - psc_linear_scan_data_idx * psc_scan_types_per_int;\n" <<
            "\n" <<
            "                        __local int *psc_src = (__local int *) &(\n" <<
            "                        psc_ldata\n" <<
            "                        [psc_linear_scan_data_idx % psc_K]\n" <<
            "                        [psc_linear_scan_data_idx / psc_K].psc_value);\n" <<
            "\n" <<
            "                        psc_dest[psc_linear_index] = psc_src[remainder];\n" <<
            "                    }\n" <<
            "                }\n" <<
            "\n" <<
            "            }\n" <<
            "\n" <<
//            "            pycl_printf(("after write\n"));\n" <<
            "\n" <<
            "            // }}}\n" <<
            "\n" <<
            "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
            "        }\n" <<
            "\n" <<
            "    // write interval psc_sum\n" <<
            "        if (get_local_id(0) == 0)\n" <<
            "        {\n" <<
            "            psc_interval_results[get_group_id(0)] = psc_partial_scan_buffer[psc_interval_end - 1];\n" <<
            "        }\n" <<
            "}    \n";


        return SourceCode(code.str(), name);
    }

    template <class BaseModel, typename RealType>
    SourceCode
    GpuModelSpecificsCox<BaseModel, RealType>::writecodeForScanLev2Kernel() {

        std::string name = "scan_lev2";

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel\n" <<
//                "__attribute__((reqd_work_group_size(psc_WG_SIZE, 1, 1)))\n" <<
                "void scan_lev2(\n" <<
                "    __global float *input_ary, __global float *output_ary, __global float *interval_sums,\n" <<
                "    __global float *restrict psc_partial_scan_buffer,\n" <<
                "    const int N,\n" <<
                "    const int psc_interval_size\n" <<
                "    )\n" <<
                "{\n" <<
                "\n" <<
                "\n" <<
                "    // index psc_K in first dimension used for psc_carry storage\n" <<
                "        struct psc_wrapped_scan_type\n" <<
                "        {\n" <<
                "            float psc_value;\n" <<
                "        };\n" <<
                "    // padded in psc_WG_SIZE to avoid bank conflicts\n" <<
                "    __local struct psc_wrapped_scan_type psc_ldata[psc_K + 1][psc_WG_SIZE + 1];\n" <<
                "\n" <<
                "    const int psc_interval_begin = psc_interval_size * get_group_id(0);\n" <<
                "    const int psc_interval_end   = min(psc_interval_begin + psc_interval_size, N);\n" <<
                "\n" <<
                "    const int psc_unit_size  = psc_K * psc_WG_SIZE;\n" <<
                "\n" <<
                "    int psc_unit_base = psc_interval_begin;\n" <<
                "\n" <<
                "\n" <<
                "            for(; psc_unit_base + psc_unit_size <= psc_interval_end; psc_unit_base += psc_unit_size)\n" <<
                "\n" <<
                "        {\n" <<
                "\n" <<
                "\n" <<
                "            // {{{ read a unit's worth of data from psc_global\n" <<
                "\n" <<
                "            for(int psc_k = 0; psc_k < psc_K; psc_k++)\n" <<
                "            {\n" <<
                "                const int psc_offset = psc_k*psc_WG_SIZE + get_local_id(0);\n" <<
                "                const int psc_read_i = psc_unit_base + psc_offset;\n" <<
                "\n" <<
                "                {\n" <<
                "\n" <<
                "                    float psc_scan_value = (interval_sums[psc_read_i]);\n" <<
                "\n" <<
                "                    const int psc_o_mod_k = psc_offset % psc_K;\n" <<
                "                    const int psc_o_div_k = psc_offset / psc_K;\n" <<
                "                    psc_ldata[psc_o_mod_k][psc_o_div_k].psc_value = psc_scan_value;\n" <<
                "\n" <<
                "                }\n" <<
                "            }\n" <<
                "\n" <<
//                "            pycl_printf(("after read from psc_global\n"));\n" <<
                "\n" <<
                "            // }}}\n" <<
                "\n" <<
                "            // {{{ psc_carry in from previous unit, if applicable\n" <<
                "\n" <<
                "\n" <<
                "            if (get_local_id(0) == 0 && psc_unit_base != psc_interval_begin)\n" <<
                "            {\n" <<
                "                float psc_tmp = psc_ldata[psc_K][psc_WG_SIZE - 1].psc_value;\n" <<
                "                float psc_tmp_aux = psc_ldata[0][0].psc_value;\n" <<
                "\n" <<
                "                psc_ldata[0][0].psc_value = psc_tmp + psc_tmp_aux;\n" <<
                "            }\n" <<
                "\n" <<
//                "            pycl_printf(("after psc_carry-in\n"));\n" <<
                "\n" <<
                "            // }}}\n" <<
                "\n" <<
                "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                "\n" <<
                "            // {{{ scan along psc_k (sequentially in each work item)\n" <<
                "\n" <<
                "            float psc_sum = psc_ldata[0][get_local_id(0)].psc_value;\n" <<
                "\n" <<
                "\n" <<
                "            for (int psc_k = 1; psc_k < psc_K; psc_k++)\n" <<
                "            {\n" <<
                "                {\n" <<
                "                    float psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;\n" <<
                "\n" <<
                "\n" <<
                "                    psc_sum = psc_sum + psc_tmp;\n" <<
                "\n" <<
                "                    psc_ldata[psc_k][get_local_id(0)].psc_value = psc_sum;\n" <<
                "                }\n" <<
                "            }\n" <<
                "\n" <<
//                "            pycl_printf(("after scan along psc_k\n"));\n" <<
                "\n" <<
                "            // }}}\n" <<
                "\n" <<
                "            // store psc_carry in out-of-bounds (padding) array entry (index psc_K) in\n" <<
                "            // the psc_K direction\n" <<
                "            psc_ldata[psc_K][get_local_id(0)].psc_value = psc_sum;\n" <<
                "\n" <<
                "\n" <<
                "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                "\n" <<
                "            // {{{ tree-based local parallel scan\n" <<
                "\n" <<
                "            // This tree-based scan works as follows:\n" <<
                "            // - Each work item adds the previous item to its current state\n" <<
                "            // - barrier\n" <<
                "            // - Each work item adds in the item from two positions to the left\n" <<
                "            // - barrier\n" <<
                "            // - Each work item adds in the item from four positions to the left\n" <<
                "            // ...\n" <<
                "            // At the end, each item has summed all prior items.\n" <<
                "\n" <<
                "            // across psc_k groups, along local id\n" <<
                "            // (uses out-of-bounds psc_k=psc_K array entry for storage)\n" <<
                "\n" <<
                "            float psc_val = psc_ldata[psc_K][get_local_id(0)].psc_value;\n" <<
                "\n" <<
                "\n" <<
                "                for (int depth = 1; depth <= psc_WG_SIZE; depth<<=1) {\n" <<
                "\n" <<
                "                    // {{{ reads from local allowed, writes to local not allowed\n" <<
                "\n" <<
                "                    if (get_local_id(0) >= depth)\n" <<
                "                    {\n" <<
                "                        float psc_tmp = psc_ldata[psc_K][get_local_id(0) - depth].psc_value;\n" <<
                "                        {\n" <<
                "                            psc_val = psc_tmp + psc_val;\n" <<
                "                        }\n" <<
                "\n" <<
                "                    }\n" <<
                "\n" <<
                "                    // }}}\n" <<
                "\n" <<
                "                    barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                "\n" <<
                "                    // {{{ writes to local allowed, reads from local not allowed\n" <<
                "\n" <<
                "                    psc_ldata[psc_K][get_local_id(0)].psc_value = psc_val;\n" <<
                "\n" <<
                "                    // }}}\n" <<
                "\n" <<
                "                    barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                "                }\n" <<
                "\n" <<
                "\n" <<
//                "            pycl_printf(("after tree scan\n"));\n" <<
                "\n" <<
                "            // }}}\n" <<
                "\n" <<
                "            // {{{ update local values\n" <<
                "\n" <<
                "            if (get_local_id(0) > 0)\n" <<
                "            {\n" <<
                "                psc_sum = psc_ldata[psc_K][get_local_id(0) - 1].psc_value;\n" <<
                "\n" <<
                "                for(int psc_k = 0; psc_k < psc_K; psc_k++)\n" <<
                "                {\n" <<
                "                    {\n" <<
                "                        float psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;\n" <<
                "                        psc_ldata[psc_k][get_local_id(0)].psc_value = psc_sum + psc_tmp;\n" <<
                "                    }\n" <<
                "                }\n" <<
                "            }\n" <<
                "\n" <<
                "\n" <<
//                "            pycl_printf(("after local update\n"));\n" <<
                "\n" <<
                "            // }}}\n" <<
                "\n" <<
                "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                "\n" <<
                "            // {{{ write data\n" <<
                "\n" <<
                "            {\n" <<
                "                // work hard with index math to achieve contiguous 32-bit stores\n" <<
                "                __global int *psc_dest =\n" <<
                "                    (__global int *) (psc_partial_scan_buffer + psc_unit_base);\n" <<
                "\n" <<
                "\n" <<
                "\n" <<
                "                const int psc_scan_types_per_int = 1;\n" <<
                "\n" <<
                "\n" <<
                "                    for (int k = 0; k < psc_K; k++)\n" <<
                "                    {\n" <<
                "                        int psc_linear_index = k*psc_WG_SIZE + get_local_id(0);\n" <<
                "                        int psc_linear_scan_data_idx =\n" <<
                "                                psc_linear_index / psc_scan_types_per_int;\n" <<
                "                        int remainder =\n" <<
                "                                psc_linear_index - psc_linear_scan_data_idx * psc_scan_types_per_int;\n" <<
                "\n" <<
                "                        __local int *psc_src = (__local int *) &(\n" <<
                "                        psc_ldata\n" <<
                "                        [psc_linear_scan_data_idx % psc_K]\n" <<
                "                        [psc_linear_scan_data_idx / psc_K].psc_value);\n" <<
                "\n" <<
                "                        psc_dest[psc_linear_index] = psc_src[remainder];\n" <<
                "                    }\n" <<
                "\n" <<
                "\n" <<
                "            }\n" <<
                "\n" <<
//                "            pycl_printf(("after write\n"));\n" <<
                "\n" <<
                "            // }}}\n" <<
                "\n" <<
                "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                "        }\n" <<
                "\n" <<
                "\n" <<
                "            if (psc_unit_base < psc_interval_end)\n" <<
                "\n" <<
                "        {\n" <<
                "\n" <<
                "            // {{{ read a unit's worth of data from psc_global\n" <<
                "\n" <<
                "            for(int psc_k = 0; psc_k < psc_K; psc_k++)\n" <<
                "            {\n" <<
                "                const int psc_offset = psc_k*psc_WG_SIZE + get_local_id(0);\n" <<
                "                const int psc_read_i = psc_unit_base + psc_offset;\n" <<
                "\n" <<
                "                if (psc_read_i < psc_interval_end)\n" <<
                "                {\n" <<
                "\n" <<
                "                    float psc_scan_value = (interval_sums[psc_read_i]);\n" <<
                "\n" <<
                "                    const int psc_o_mod_k = psc_offset % psc_K;\n" <<
                "                    const int psc_o_div_k = psc_offset / psc_K;\n" <<
                "                    psc_ldata[psc_o_mod_k][psc_o_div_k].psc_value = psc_scan_value;\n" <<
                "\n" <<
                "                }\n" <<
                "            }\n" <<
                "\n" <<
//                "            pycl_printf(("after read from psc_global\n"));\n" <<
                "\n" <<
                "            // }}}\n" <<
                "\n" <<
                "            // {{{ psc_carry in from previous unit, if applicable\n" <<
                "\n" <<
                "\n" <<
                "            if (get_local_id(0) == 0 && psc_unit_base != psc_interval_begin)\n" <<
                "            {\n" <<
                "                float psc_tmp = psc_ldata[psc_K][psc_WG_SIZE - 1].psc_value;\n" <<
                "                float psc_tmp_aux = psc_ldata[0][0].psc_value;\n" <<
                "\n" <<
                "                psc_ldata[0][0].psc_value = psc_tmp + psc_tmp_aux;\n" <<
                "            }\n" <<
                "\n" <<
//                "            pycl_printf(("after psc_carry-in\n"));\n" <<
                "\n" <<
                "            // }}}\n" <<
                "\n" <<
                "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                "\n" <<
                "            // {{{ scan along psc_k (sequentially in each work item)\n" <<
                "\n" <<
                "            float psc_sum = psc_ldata[0][get_local_id(0)].psc_value;\n" <<
                "\n" <<
                "                const int psc_offset_end = psc_interval_end - psc_unit_base;\n" <<
                "\n" <<
                "            for (int psc_k = 1; psc_k < psc_K; psc_k++)\n" <<
                "            {\n" <<
                "                if ((int) (psc_K * get_local_id(0) + psc_k) < psc_offset_end)\n" <<
                "                {\n" <<
                "                    float psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;\n" <<
                "\n" <<
                "\n" <<
                "                    psc_sum = psc_sum + psc_tmp;\n" <<
                "\n" <<
                "                    psc_ldata[psc_k][get_local_id(0)].psc_value = psc_sum;\n" <<
                "                }\n" <<
                "            }\n" <<
                "\n" <<
//                "            pycl_printf(("after scan along psc_k\n"));\n" <<
                "\n" <<
                "            // }}}\n" <<
                "\n" <<
                "            // store psc_carry in out-of-bounds (padding) array entry (index psc_K) in\n" <<
                "            // the psc_K direction\n" <<
                "            psc_ldata[psc_K][get_local_id(0)].psc_value = psc_sum;\n" <<
                "\n" <<
                "\n" <<
                "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                "\n" <<
                "            // {{{ tree-based local parallel scan\n" <<
                "\n" <<
                "            // This tree-based scan works as follows:\n" <<
                "            // - Each work item adds the previous item to its current state\n" <<
                "            // - barrier\n" <<
                "            // - Each work item adds in the item from two positions to the left\n" <<
                "            // - barrier\n" <<
                "            // - Each work item adds in the item from four positions to the left\n" <<
                "            // ...\n" <<
                "            // At the end, each item has summed all prior items.\n" <<
                "\n" <<
                "            // across psc_k groups, along local id\n" <<
                "            // (uses out-of-bounds psc_k=psc_K array entry for storage)\n" <<
                "\n" <<
                "            float psc_val = psc_ldata[psc_K][get_local_id(0)].psc_value;\n" <<
                "\n" <<
                "\n" <<
                "                for (int depth = 1; depth <= psc_WG_SIZE; depth<<=1) {\n" <<
                "\n" <<
                "                    // {{{ reads from local allowed, writes to local not allowed\n" <<
                "\n" <<
                "                    if (get_local_id(0) >= depth)\n" <<
                "                    {\n" <<
                "                        float psc_tmp = psc_ldata[psc_K][get_local_id(0) - depth].psc_value;\n" <<
                "                        if (psc_K*get_local_id(0) < psc_offset_end)\n" <<
                "                        {\n" <<
                "                            psc_val = psc_tmp + psc_val;\n" <<
                "                        }\n" <<
                "\n" <<
                "                    }\n" <<
                "\n" <<
                "                    // }}}\n" <<
                "\n" <<
                "                    barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                "\n" <<
                "                    // {{{ writes to local allowed, reads from local not allowed\n" <<
                "\n" <<
                "                    psc_ldata[psc_K][get_local_id(0)].psc_value = psc_val;\n" <<
                "\n" <<
                "                    // }}}\n" <<
                "\n" <<
                "                    barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                "                }\n" <<
                "\n" <<
//                "            pycl_printf(("after tree scan\n"));\n" <<
                "\n" <<
                "            // }}}\n" <<
                "\n" <<
                "            // {{{ update local values\n" <<
                "\n" <<
                "            if (get_local_id(0) > 0)\n" <<
                "            {\n" <<
                "                psc_sum = psc_ldata[psc_K][get_local_id(0) - 1].psc_value;\n" <<
                "\n" <<
                "                for(int psc_k = 0; psc_k < psc_K; psc_k++)\n" <<
                "                {\n" <<
                "                    if (psc_K * get_local_id(0) + psc_k < psc_offset_end)\n" <<
                "                    {\n" <<
                "                        float psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;\n" <<
                "                        psc_ldata[psc_k][get_local_id(0)].psc_value = psc_sum + psc_tmp;\n" <<
                "                    }\n" <<
                "                }\n" <<
                "            }\n" <<
                "\n" <<
                "\n" <<
//                "            pycl_printf(("after local update\n"));\n" <<
                "\n" <<
                "            // }}}\n" <<
                "\n" <<
                "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                "\n" <<
                "            // {{{ write data\n" <<
                "\n" <<
                "            {\n" <<
                "                // work hard with index math to achieve contiguous 32-bit stores\n" <<
                "                __global int *psc_dest =\n" <<
                "                    (__global int *) (psc_partial_scan_buffer + psc_unit_base);\n" <<
                "\n" <<
                "\n" <<
                "\n" <<
                "                const int psc_scan_types_per_int = 1;\n" <<
                "\n" <<
                "\n" <<
                "                for (int k = 0; k < psc_K; k++) {\n" <<
                "\n" <<
                "                    if (k*psc_WG_SIZE + get_local_id(0) < psc_scan_types_per_int*(psc_interval_end - psc_unit_base))\n" <<
                "                    {\n" <<
                "                        int psc_linear_index = k*psc_WG_SIZE + get_local_id(0);\n" <<
                "                        int psc_linear_scan_data_idx =\n" <<
                "                                psc_linear_index / psc_scan_types_per_int;\n" <<
                "                        int remainder =\n" <<
                "                                psc_linear_index - psc_linear_scan_data_idx * psc_scan_types_per_int;\n" <<
                "\n" <<
                "                        __local int *psc_src = (__local int *) &(\n" <<
                "                        psc_ldata\n" <<
                "                        [psc_linear_scan_data_idx % psc_K]\n" <<
                "                        [psc_linear_scan_data_idx / psc_K].psc_value);\n" <<
                "\n" <<
                "                        psc_dest[psc_linear_index] = psc_src[remainder];\n" <<
                "                    }\n" <<
                "                }\n" <<
                "\n" <<
                "            }\n" <<
                "\n" <<
//                "            pycl_printf(("after write\n"));\n" <<
                "\n" <<
                "            // }}}\n" <<
                "\n" <<
                "            barrier(CLK_LOCAL_MEM_FENCE);\n" <<
                "        }\n" <<
                "    }\n";

        return SourceCode(code.str(), name);
    }

    template <class BaseModel, typename RealType>
    SourceCode
    GpuModelSpecificsCox<BaseModel, RealType>::writecodeForScanUpdKernel() {

        std::string name = "scan_final_update";

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel\n" <<
//             "__attribute__((reqd_work_group_size(psc_WG_SIZE, 1, 1)))\n" <<
             "void scan_final_update(\n" <<
             "    __global float *input_ary, __global float *output_ary,\n" <<
             "    const int N,\n" <<
             "    const int psc_interval_size,\n" <<
             "    __global float *restrict psc_interval_results,\n" <<
             "    __global float *restrict psc_partial_scan_buffer\n" <<
             "    )\n" <<
             "{\n" <<
             "    const int psc_interval_begin = psc_interval_size * get_group_id(0);\n" <<
             "    const int psc_interval_end = min(psc_interval_begin + psc_interval_size, N);\n" <<
             "\n" <<
             "    // psc_carry from last interval\n" <<
             "    float psc_carry = 0;\n" <<
             "    if (get_group_id(0) != 0)\n" <<
             "        psc_carry = psc_interval_results[get_group_id(0) - 1];\n" <<
             "\n" <<
             "        // {{{ no look-behind ('prev_item' not in output_statement -> simpler)\n" <<
             "        int psc_update_i = psc_interval_begin+get_local_id(0);\n" <<
             "        for(; psc_update_i < psc_interval_end; psc_update_i += psc_WG_SIZE)\n" <<
             "        {\n" <<
             "            float psc_partial_val = psc_partial_scan_buffer[psc_update_i];\n" <<
             "            float item = psc_carry + psc_partial_val;\n" <<
             "            int i = psc_update_i;\n" <<
             "            { output_ary[i] = item;; }\n" <<
             "        }\n" <<
             "        // }}}\n" <<
             "    }\n";

        return SourceCode(code.str(), name);
    }

    template <class BaseModel, typename RealType>
    SourceCode
    GpuModelSpecificsCox<BaseModel, RealType>::writeCodeForComputeAccumlatedDenominatorKernel(bool useWeights) {

        std::string name = "computeAccumlatedDenominator";

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(          \n" <<
//             "       const uint offX,           \n" <<
//             "       const uint offK,           \n" <<
                "       const uint taskCount,           \n" <<
                "       __global REAL* denominator,     \n" <<
                "       __global REAL* accDenominator,  \n" <<
                "       __global REAL* buffer,          \n" <<
                "       __global const int* id)  {      \n" ;

        // Initialization
        code << "   uint gid = get_global_id(0);   \n" <<
                "   uint lid = get_local_id(0);    \n" <<
                "   const uint scale = taskCount;  \n" <<
                "   uint d = 1;                    \n" ;

        code << "   barrier(CLK_LOCAL_MEM_FENCE);                   \n" ;
        code << "   buffer[lid] = denominator[gid];             \n" ;

        // up-sweep
        code << "   for(uint s = (scale >> 1); s > 0; s >>= 1) {    \n" <<
                "       barrier(CLK_LOCAL_MEM_FENCE);               \n" <<
                "       if (lid < s) {                              \n" <<
                "           uint i = d*(2*lid+1)-1;                 \n" <<
                "           uint j = d*(2*lid+2)-1;                 \n" <<
                "           buffer[j] += buffer[i];                 \n" <<
                "       }                                           \n" <<
                "       d <<= 1;                                    \n" <<
                "   }                                               \n";

        // down-sweep
        code << "   if(lid == 0) buffer[taskCount - 1] = 0;     \n" ;
//        code << "   if(lid == 0) {                              \n" <<
//                "       const REAL sum = buffer[taskCount - 1];       \n" <<
//                "       accDenominator[taskCount - 1] = buffer[taskCount - 1];       \n" <<
//                "       buffer[scale - 1] = 0;              \n" <<
//                "   }                                           \n";
        code << "   for(uint s = 1; s < scale; s <<= 1) {       \n" <<
                "       d >>= 1;                                \n" <<
                "       barrier(CLK_LOCAL_MEM_FENCE);           \n" <<
                "       if (lid < s) {                          \n" <<
                "           uint i = d*(2*lid+1)-1;             \n" <<
                "           uint j = d*(2*lid+2)-1;             \n" <<
                "           REAL temp = buffer[i];              \n" <<
                "           buffer[i] = buffer[j];              \n" <<
                "           buffer[j] += temp;                  \n" <<
                "       }                                       \n" <<
                "   }                                           \n";

        // copy results
//        code << "   accDenominator[taskCount - 1] = sum;                \n";
        code << "   barrier(CLK_LOCAL_MEM_FENCE);                   \n" <<
                "   if (lid < scale) {                              \n" <<
                "       accDenominator[gid] = buffer[lid+1];        \n" <<
                "   }                                               \n";

        code << "}    \n";

        return SourceCode(code.str(), name);
    }

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
