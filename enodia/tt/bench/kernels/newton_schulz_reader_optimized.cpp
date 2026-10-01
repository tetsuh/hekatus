// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

namespace {
constexpr std::uint32_t cb_r_real = 0;
constexpr std::uint32_t cb_r_negative_imag = 1;
constexpr std::uint32_t cb_r_imag = 2;
constexpr std::uint32_t cb_x0_real = 3;
constexpr std::uint32_t cb_x0_imag = 4;
constexpr std::uint32_t cb_identity = 5;
constexpr std::uint32_t cb_zero = 6;
constexpr std::uint32_t cb_r_negative_real = 14;

template <typename Accessor>
void read_one(std::uint32_t cb, std::uint32_t tile_id,
              const Accessor &accessor) {
  cb_reserve_back(cb, 1);
  noc_async_read_page(tile_id, accessor, get_write_ptr(cb));
  noc_async_read_barrier();
  cb_push_back(cb, 1);
}

template <bool fuse_s, bool batch_reads, typename Accessor>
void read_matrix(std::uint32_t tile_id, const Accessor &r_real,
                 const Accessor &r_negative_imag, const Accessor &r_imag,
                 const Accessor &x0_real, const Accessor &x0_imag,
                 const Accessor &r_negative_real) {
  if constexpr (batch_reads) {
    // Reserve every destination before issuing any DMA.  This prevents a
    // later reservation from overtaking an earlier read while the compute
    // kernel holds resident R pages across all eight iterations.
    if constexpr (!fuse_s) {
      cb_reserve_back(cb_r_real, 1);
    }
    cb_reserve_back(cb_r_negative_imag, 1);
    cb_reserve_back(cb_r_imag, 1);
    if constexpr (fuse_s) {
      cb_reserve_back(cb_r_negative_real, 1);
    }
    cb_reserve_back(cb_x0_real, 1);
    cb_reserve_back(cb_x0_imag, 1);

    if constexpr (!fuse_s) {
      noc_async_read_page(tile_id, r_real, get_write_ptr(cb_r_real));
    }
    noc_async_read_page(tile_id, r_negative_imag,
                        get_write_ptr(cb_r_negative_imag));
    noc_async_read_page(tile_id, r_imag, get_write_ptr(cb_r_imag));
    if constexpr (fuse_s) {
      noc_async_read_page(tile_id, r_negative_real,
                          get_write_ptr(cb_r_negative_real));
    }
    noc_async_read_page(tile_id, x0_real, get_write_ptr(cb_x0_real));
    noc_async_read_page(tile_id, x0_imag, get_write_ptr(cb_x0_imag));
    noc_async_read_barrier();

    if constexpr (!fuse_s) {
      cb_push_back(cb_r_real, 1);
    }
    cb_push_back(cb_r_negative_imag, 1);
    cb_push_back(cb_r_imag, 1);
    if constexpr (fuse_s) {
      cb_push_back(cb_r_negative_real, 1);
    }
    cb_push_back(cb_x0_real, 1);
    cb_push_back(cb_x0_imag, 1);
  } else {
    if constexpr (!fuse_s) {
      read_one(cb_r_real, tile_id, r_real);
    }
    read_one(cb_r_negative_imag, tile_id, r_negative_imag);
    read_one(cb_r_imag, tile_id, r_imag);
    if constexpr (fuse_s) {
      read_one(cb_r_negative_real, tile_id, r_negative_real);
    }
    read_one(cb_x0_real, tile_id, x0_real);
    read_one(cb_x0_imag, tile_id, x0_imag);
  }
}

template <bool fuse_s, bool batch_reads, typename Accessor>
void read_matrix_block(
    std::uint32_t tile_id,
    std::uint32_t block_count,
    const Accessor &r_real,
    const Accessor &r_negative_imag,
    const Accessor &r_imag,
    const Accessor &x0_real,
    const Accessor &x0_imag,
    const Accessor &r_negative_real) {
  if constexpr (!fuse_s) {
    cb_reserve_back(cb_r_real, block_count);
  }
  cb_reserve_back(cb_r_negative_imag, block_count);
  cb_reserve_back(cb_r_imag, block_count);
  if constexpr (fuse_s) {
    cb_reserve_back(cb_r_negative_real, block_count);
  }
  cb_reserve_back(cb_x0_real, block_count);
  cb_reserve_back(cb_x0_imag, block_count);

  for (std::uint32_t index = 0; index < block_count; ++index) {
    const std::uint32_t tile = tile_id + index;
    const std::uint32_t r_negative_imag_ptr =
        get_write_ptr(cb_r_negative_imag) + index * get_tile_size(cb_r_negative_imag);
    const std::uint32_t r_imag_ptr = get_write_ptr(cb_r_imag) + index * get_tile_size(cb_r_imag);
    const std::uint32_t x0_real_ptr = get_write_ptr(cb_x0_real) + index * get_tile_size(cb_x0_real);
    const std::uint32_t x0_imag_ptr = get_write_ptr(cb_x0_imag) + index * get_tile_size(cb_x0_imag);
    if constexpr (batch_reads) {
      if constexpr (!fuse_s) {
        const std::uint32_t r_real_ptr =
            get_write_ptr(cb_r_real) + index * get_tile_size(cb_r_real);
        noc_async_read_page(tile, r_real, r_real_ptr);
      }
      noc_async_read_page(tile, r_negative_imag, r_negative_imag_ptr);
      noc_async_read_page(tile, r_imag, r_imag_ptr);
      if constexpr (fuse_s) {
        const std::uint32_t r_negative_real_ptr =
            get_write_ptr(cb_r_negative_real) + index * get_tile_size(cb_r_negative_real);
        noc_async_read_page(tile, r_negative_real, r_negative_real_ptr);
      }
      noc_async_read_page(tile, x0_real, x0_real_ptr);
      noc_async_read_page(tile, x0_imag, x0_imag_ptr);
    } else {
      if constexpr (!fuse_s) {
        const std::uint32_t r_real_ptr =
            get_write_ptr(cb_r_real) + index * get_tile_size(cb_r_real);
        noc_async_read_page(tile, r_real, r_real_ptr);
        noc_async_read_barrier();
      }
      noc_async_read_page(tile, r_negative_imag, r_negative_imag_ptr);
      noc_async_read_barrier();
      noc_async_read_page(tile, r_imag, r_imag_ptr);
      noc_async_read_barrier();
      if constexpr (fuse_s) {
        const std::uint32_t r_negative_real_ptr =
            get_write_ptr(cb_r_negative_real) + index * get_tile_size(cb_r_negative_real);
        noc_async_read_page(tile, r_negative_real, r_negative_real_ptr);
        noc_async_read_barrier();
      }
      noc_async_read_page(tile, x0_real, x0_real_ptr);
      noc_async_read_barrier();
      noc_async_read_page(tile, x0_imag, x0_imag_ptr);
      noc_async_read_barrier();
    }
  }
  if constexpr (batch_reads) {
    noc_async_read_barrier();
  }

  if constexpr (!fuse_s) {
    cb_push_back(cb_r_real, block_count);
  }
  cb_push_back(cb_r_negative_imag, block_count);
  cb_push_back(cb_r_imag, block_count);
  if constexpr (fuse_s) {
    cb_push_back(cb_r_negative_real, block_count);
  }
  cb_push_back(cb_x0_real, block_count);
  cb_push_back(cb_x0_imag, block_count);
}
} // namespace

void kernel_main() {
  // Runtime addresses follow the host tensor order: R variants, both X0
  // halves, then resident constants.  TensorAccessor preserves the selected
  // placement for each address, so fused S still omits positive R-real.
  constexpr std::uint32_t iterations = get_compile_time_arg_val(0);
  constexpr bool fuse_s = get_compile_time_arg_val(1) != 0;
  constexpr bool batch_reads = get_compile_time_arg_val(2) != 0;
  constexpr std::uint32_t matrix_block = get_compile_time_arg_val(3);
  (void)iterations;

  std::uint32_t r_real_address = 0;
  std::uint32_t r_negative_imag_address;
  std::uint32_t r_imag_address;
  std::uint32_t r_negative_real_address = 0;
  std::uint32_t x0_real_address;
  std::uint32_t x0_imag_address;
  std::uint32_t identity_address;
  std::uint32_t zero_address;
  std::uint32_t start_tile;
  std::uint32_t tile_count;
  if constexpr (fuse_s) {
    // Fused S has no positive R-real runtime argument.  Keep the signed
    // inputs contiguous so their accessor offsets match the host tensor list.
    r_negative_imag_address = get_arg_val<std::uint32_t>(0);
    r_imag_address = get_arg_val<std::uint32_t>(1);
    r_negative_real_address = get_arg_val<std::uint32_t>(2);
    x0_real_address = get_arg_val<std::uint32_t>(3);
    x0_imag_address = get_arg_val<std::uint32_t>(4);
    identity_address = get_arg_val<std::uint32_t>(5);
    zero_address = get_arg_val<std::uint32_t>(6);
    start_tile = get_arg_val<std::uint32_t>(7);
    tile_count = get_arg_val<std::uint32_t>(8);
  } else {
    r_real_address = get_arg_val<std::uint32_t>(0);
    r_negative_imag_address = get_arg_val<std::uint32_t>(1);
    r_imag_address = get_arg_val<std::uint32_t>(2);
    x0_real_address = get_arg_val<std::uint32_t>(3);
    x0_imag_address = get_arg_val<std::uint32_t>(4);
    identity_address = get_arg_val<std::uint32_t>(5);
    zero_address = get_arg_val<std::uint32_t>(6);
    start_tile = get_arg_val<std::uint32_t>(7);
    tile_count = get_arg_val<std::uint32_t>(8);
  }

  constexpr auto first_input_args = TensorAccessorArgs<4>();
  constexpr auto r_negative_imag_args = TensorAccessorArgs<
      (fuse_s ? 4 : first_input_args.next_compile_time_args_offset())>();
  constexpr auto r_imag_args = TensorAccessorArgs<
      r_negative_imag_args.next_compile_time_args_offset()>();
  constexpr auto r_negative_real_args =
      TensorAccessorArgs<r_imag_args.next_compile_time_args_offset()>();
  constexpr auto x0_real_args = TensorAccessorArgs<
      (fuse_s ? r_negative_real_args.next_compile_time_args_offset()
               : r_imag_args.next_compile_time_args_offset())>();
  constexpr auto x0_imag_args =
      TensorAccessorArgs<x0_real_args.next_compile_time_args_offset()>();
  constexpr auto identity_args =
      TensorAccessorArgs<x0_imag_args.next_compile_time_args_offset()>();
  constexpr auto zero_args =
      TensorAccessorArgs<identity_args.next_compile_time_args_offset()>();

  const auto r_negative_imag =
      TensorAccessor(r_negative_imag_args, r_negative_imag_address);
  const auto r_imag = TensorAccessor(r_imag_args, r_imag_address);
  const auto x0_real = TensorAccessor(x0_real_args, x0_real_address);
  const auto x0_imag = TensorAccessor(x0_imag_args, x0_imag_address);
  const auto identity = TensorAccessor(identity_args, identity_address);
  const auto zero = TensorAccessor(zero_args, zero_address);

  // Constants are read first and retained at the front of their CBs.  Every
  // matrix then publishes external inputs in compute consumption order.
  read_one(cb_identity, 0, identity);
  read_one(cb_zero, 0, zero);
  if constexpr (fuse_s) {
    const auto r_negative_real =
        TensorAccessor(r_negative_real_args, r_negative_real_address);
    if constexpr (matrix_block == 1) {
      for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        // The first argument is a deliberately unused template placeholder;
        // fused S never references CB_R_REAL or its accessor.
        read_matrix<fuse_s, batch_reads>(start_tile + offset, r_negative_imag,
                                         r_negative_imag, r_imag, x0_real,
                                         x0_imag, r_negative_real);
      }
    } else {
      for (std::uint32_t offset = 0; offset < tile_count; offset += matrix_block) {
        const std::uint32_t block_count =
            (tile_count - offset < matrix_block) ? (tile_count - offset) : matrix_block;
        read_matrix_block<fuse_s, batch_reads>(
            start_tile + offset, block_count, r_negative_imag,
            r_negative_imag, r_imag, x0_real, x0_imag, r_negative_real);
      }
    }
  } else {
    const auto r_real = TensorAccessor(first_input_args, r_real_address);
    if constexpr (matrix_block == 1) {
      for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        read_matrix<fuse_s, batch_reads>(start_tile + offset, r_real,
                                         r_negative_imag, r_imag, x0_real,
                                         x0_imag, r_real);
      }
    } else {
      for (std::uint32_t offset = 0; offset < tile_count; offset += matrix_block) {
        const std::uint32_t block_count =
            (tile_count - offset < matrix_block) ? (tile_count - offset) : matrix_block;
        read_matrix_block<fuse_s, batch_reads>(
            start_tile + offset, block_count, r_real, r_negative_imag,
            r_imag, x0_real, x0_imag, r_real);
      }
    }
  }
}
