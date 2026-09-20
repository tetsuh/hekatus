// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr std::uint32_t cb_matmul_a = 0;
constexpr std::uint32_t cb_matmul_b = 1;
constexpr std::uint32_t cb_product_rr = 2;
constexpr std::uint32_t cb_product_ii = 3;
constexpr std::uint32_t cb_product_ri = 4;
constexpr std::uint32_t cb_product_ir = 5;
constexpr std::uint32_t cb_identity = 6;
constexpr std::uint32_t cb_zero = 7;
constexpr std::uint32_t cb_matmul_product = 16;
constexpr std::uint32_t cb_rx_real = 17;
constexpr std::uint32_t cb_rx_imag = 18;
constexpr std::uint32_t cb_s_real = 19;
constexpr std::uint32_t cb_s_imag = 20;
constexpr std::uint32_t cb_state_real = 21;
constexpr std::uint32_t cb_state_imag = 22;
constexpr std::uint32_t cb_out_real = 23;
constexpr std::uint32_t cb_out_imag = 24;

template <std::uint32_t output>
void pack_one() {
    tile_regs_wait();
    pack_tile(0, output);
    tile_regs_release();
    cb_push_back(output, 1);
}

void matmul_one() {
    cb_wait_front(cb_matmul_a, 1);
    cb_wait_front(cb_matmul_b, 1);
    cb_reserve_back(cb_matmul_product, 1);
    tile_regs_acquire();
    matmul_tiles(cb_matmul_a, cb_matmul_b, 0, 0, 0);
    tile_regs_commit();
    pack_one<cb_matmul_product>();
    cb_pop_front(cb_matmul_a, 1);
    cb_pop_front(cb_matmul_b, 1);
}

template <std::uint32_t input_a, std::uint32_t input_b, std::uint32_t output>
void add_one() {
    cb_wait_front(input_a, 1);
    cb_wait_front(input_b, 1);
    binary_op_init_common(input_a, input_b, output);
    add_tiles_init(input_a, input_b);
    cb_reserve_back(output, 1);
    tile_regs_acquire();
    add_tiles(input_a, input_b, 0, 0, 0);
    tile_regs_commit();
    pack_one<output>();
    cb_pop_front(input_a, 1);
    cb_pop_front(input_b, 1);
}

template <std::uint32_t input_a, std::uint32_t input_b, std::uint32_t output>
void subtract_one() {
    cb_wait_front(input_a, 1);
    cb_wait_front(input_b, 1);
    binary_op_init_common(input_a, input_b, output);
    sub_tiles_init(input_a, input_b);
    cb_reserve_back(output, 1);
    tile_regs_acquire();
    sub_tiles(input_a, input_b, 0, 0, 0);
    tile_regs_commit();
    pack_one<output>();
    cb_pop_front(input_a, 1);
    cb_pop_front(input_b, 1);
}

template <std::uint32_t out_real, std::uint32_t out_imag>
void complex_matmul() {
    matmul_init(cb_matmul_a, cb_matmul_b);
    matmul_one();
    matmul_one();
    matmul_one();
    matmul_one();
    subtract_one<cb_product_rr, cb_product_ii, out_real>();
    add_one<cb_product_ri, cb_product_ir, out_imag>();
}
}  // namespace

void kernel_main() {
    constexpr std::uint32_t iterations = get_compile_time_arg_val(0);
    constexpr std::uint32_t tile_count = get_compile_time_arg_val(1);
    static_assert(iterations > 0, "the fixed iteration count must be positive");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_matmul_a, cb_matmul_b, cb_matmul_product);
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            complex_matmul<cb_rx_real, cb_rx_imag>();
            subtract_one<cb_identity, cb_product_rr, cb_s_real>();
            subtract_one<cb_zero, cb_product_ii, cb_s_imag>();
            if (iteration + 1 == iterations) {
                complex_matmul<cb_out_real, cb_out_imag>();
            } else {
                complex_matmul<cb_state_real, cb_state_imag>();
            }
        }
    }
}
