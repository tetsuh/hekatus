// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_r_real = 0;
constexpr std::uint32_t cb_r_negative_imag = 1;
constexpr std::uint32_t cb_r_imag = 2;
constexpr std::uint32_t cb_x0_real = 3;
constexpr std::uint32_t cb_x0_imag = 4;
constexpr std::uint32_t cb_identity = 5;
constexpr std::uint32_t cb_zero = 6;
constexpr std::uint32_t cb_r_negative_real = 14;
constexpr std::uint32_t cb_profile_reader = 17;
constexpr std::uint32_t profile_magic = 0x5052464C;
constexpr std::uint32_t profile_total_offset = 0;
constexpr std::uint32_t profile_cb_wait_offset = 1;
constexpr std::uint32_t profile_noc_read_offset = 2;
constexpr std::uint32_t profile_section_sum_offset = 8;
constexpr std::uint32_t profile_residual_offset = 9;
constexpr std::uint32_t profile_event_count_offset = 10;
constexpr std::uint32_t profile_warmup_event_count_offset = 11;
constexpr std::uint32_t profile_warmup_base = 32;
constexpr std::uint32_t profile_ready_offset = 31;
constexpr std::uint32_t profile_warmup_ready_offset = 63;
constexpr std::uint32_t profile_words = 32 * 32;

struct ProfileCounters {
    std::uint32_t total_start = 0;
    std::uint32_t total_end = 0;
    std::uint32_t warmup_start = 0;
    std::uint32_t warmup_end = 0;
    std::uint32_t event_count = 0;
    std::uint32_t warmup_event_count = 0;
    std::uint32_t cb_wait = 0;
    std::uint32_t noc_read = 0;
    std::uint32_t warmup_cb_wait = 0;
    std::uint32_t warmup_noc_read = 0;
};

void add_profile_cycles(std::uint32_t& total, std::uint32_t& warmup, std::uint32_t cycles, bool is_warmup) {
    total += cycles;
    if (is_warmup) {
        warmup += cycles;
    }
}

template <typename Accessor>
void read_tile(std::uint32_t cb, std::uint32_t tile_id, const Accessor& accessor) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(tile_id, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}

template <typename Accessor>
void read_tile_profiled(
    std::uint32_t cb,
    std::uint32_t tile_id,
    const Accessor& accessor,
    ProfileCounters& counters,
    bool warmup) {
    const std::uint32_t wait_start = get_timestamp_32b();
    cb_reserve_back(cb, 1);
    add_profile_cycles(
        counters.cb_wait,
        counters.warmup_cb_wait,
        get_timestamp_32b() - wait_start,
        warmup);

    const std::uint32_t read_start = get_timestamp_32b();
    noc_async_read_page(tile_id, accessor, get_write_ptr(cb));
    noc_async_read_barrier();
    add_profile_cycles(
        counters.noc_read,
        counters.warmup_noc_read,
        get_timestamp_32b() - read_start,
        warmup);
    cb_push_back(cb, 1);
    ++counters.event_count;
    if (warmup) {
        ++counters.warmup_event_count;
    }
}

template <bool fuse_s, bool batch_reads, typename RAccessor, typename X0Accessor>
void read_matrix(
    std::uint32_t tile_id,
    const RAccessor& r_real,
    const RAccessor& r_negative_imag,
    const RAccessor& r_imag,
    const X0Accessor& x0_real,
    const X0Accessor& x0_imag,
    const RAccessor& r_negative_real) {
    if constexpr (batch_reads) {
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
        noc_async_read_page(tile_id, r_negative_imag, get_write_ptr(cb_r_negative_imag));
        noc_async_read_page(tile_id, r_imag, get_write_ptr(cb_r_imag));
        if constexpr (fuse_s) {
            noc_async_read_page(tile_id, r_negative_real, get_write_ptr(cb_r_negative_real));
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
            read_tile(cb_r_real, tile_id, r_real);
        }
        read_tile(cb_r_negative_imag, tile_id, r_negative_imag);
        read_tile(cb_r_imag, tile_id, r_imag);
        if constexpr (fuse_s) {
            read_tile(cb_r_negative_real, tile_id, r_negative_real);
        }
        read_tile(cb_x0_real, tile_id, x0_real);
        read_tile(cb_x0_imag, tile_id, x0_imag);
    }
}

template <bool fuse_s, typename RAccessor, typename X0Accessor>
void read_matrix_profiled(
    std::uint32_t tile_id,
    const RAccessor& r_real,
    const RAccessor& r_negative_imag,
    const RAccessor& r_imag,
    const X0Accessor& x0_real,
    const X0Accessor& x0_imag,
    const RAccessor& r_negative_real,
    ProfileCounters& counters,
    bool warmup,
    bool include_x0 = true) {
    const std::uint32_t wait_start = get_timestamp_32b();
    if constexpr (!fuse_s) {
        cb_reserve_back(cb_r_real, 1);
    }
    cb_reserve_back(cb_r_negative_imag, 1);
    cb_reserve_back(cb_r_imag, 1);
    if constexpr (fuse_s) {
        cb_reserve_back(cb_r_negative_real, 1);
    }
    if (include_x0) {
        cb_reserve_back(cb_x0_real, 1);
        cb_reserve_back(cb_x0_imag, 1);
    }
    add_profile_cycles(
        counters.cb_wait,
        counters.warmup_cb_wait,
        get_timestamp_32b() - wait_start,
        warmup);

    const std::uint32_t read_start = get_timestamp_32b();
    if constexpr (!fuse_s) {
        noc_async_read_page(tile_id, r_real, get_write_ptr(cb_r_real));
    }
    noc_async_read_page(tile_id, r_negative_imag, get_write_ptr(cb_r_negative_imag));
    noc_async_read_page(tile_id, r_imag, get_write_ptr(cb_r_imag));
    if constexpr (fuse_s) {
        noc_async_read_page(tile_id, r_negative_real, get_write_ptr(cb_r_negative_real));
    }
    if (include_x0) {
        noc_async_read_page(tile_id, x0_real, get_write_ptr(cb_x0_real));
        noc_async_read_page(tile_id, x0_imag, get_write_ptr(cb_x0_imag));
    }
    noc_async_read_barrier();
    add_profile_cycles(
        counters.noc_read,
        counters.warmup_noc_read,
        get_timestamp_32b() - read_start,
        warmup);
    if constexpr (!fuse_s) {
        cb_push_back(cb_r_real, 1);
    }
    cb_push_back(cb_r_negative_imag, 1);
    cb_push_back(cb_r_imag, 1);
    if constexpr (fuse_s) {
        cb_push_back(cb_r_negative_real, 1);
    }
    if (include_x0) {
        cb_push_back(cb_x0_real, 1);
        cb_push_back(cb_x0_imag, 1);
    }
    constexpr std::uint32_t read_count = 3;
    counters.event_count += read_count + (include_x0 ? 2 : 0);
    if (warmup) {
        counters.warmup_event_count += read_count + (include_x0 ? 2 : 0);
    }
}

template <bool fuse_s, bool batch_reads, typename RAccessor, typename X0Accessor>
void read_matrix_block(
    std::uint32_t tile_id,
    std::uint32_t block_count,
    const RAccessor& r_real,
    const RAccessor& r_negative_imag,
    const RAccessor& r_imag,
    const X0Accessor& x0_real,
    const X0Accessor& x0_imag,
    const RAccessor& r_negative_real) {
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
        const std::uint32_t r_imag_ptr =
            get_write_ptr(cb_r_imag) + index * get_tile_size(cb_r_imag);
        const std::uint32_t x0_real_ptr =
            get_write_ptr(cb_x0_real) + index * get_tile_size(cb_x0_real);
        const std::uint32_t x0_imag_ptr =
            get_write_ptr(cb_x0_imag) + index * get_tile_size(cb_x0_imag);
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

template <bool fuse_s, bool batch_reads, typename RAccessor, typename X0Accessor>
void read_matrix_block_profiled(
    std::uint32_t tile_id,
    std::uint32_t block_count,
    const RAccessor& r_real,
    const RAccessor& r_negative_imag,
    const RAccessor& r_imag,
    const X0Accessor& x0_real,
    const X0Accessor& x0_imag,
    const RAccessor& r_negative_real,
    ProfileCounters& counters,
    bool warmup,
    bool include_x0 = true) {
    const std::uint32_t wait_start = get_timestamp_32b();
    if constexpr (!fuse_s) {
        cb_reserve_back(cb_r_real, block_count);
    }
    cb_reserve_back(cb_r_negative_imag, block_count);
    cb_reserve_back(cb_r_imag, block_count);
    if constexpr (fuse_s) {
        cb_reserve_back(cb_r_negative_real, block_count);
    }
    if (include_x0) {
        cb_reserve_back(cb_x0_real, block_count);
        cb_reserve_back(cb_x0_imag, block_count);
    }
    add_profile_cycles(
        counters.cb_wait,
        counters.warmup_cb_wait,
        get_timestamp_32b() - wait_start,
        warmup);

    const std::uint32_t read_start = get_timestamp_32b();
    for (std::uint32_t index = 0; index < block_count; ++index) {
        const std::uint32_t tile = tile_id + index;
        const std::uint32_t r_negative_imag_ptr =
            get_write_ptr(cb_r_negative_imag) + index * get_tile_size(cb_r_negative_imag);
        const std::uint32_t r_imag_ptr =
            get_write_ptr(cb_r_imag) + index * get_tile_size(cb_r_imag);
        const std::uint32_t x0_real_ptr =
            get_write_ptr(cb_x0_real) + index * get_tile_size(cb_x0_real);
        const std::uint32_t x0_imag_ptr =
            get_write_ptr(cb_x0_imag) + index * get_tile_size(cb_x0_imag);
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
            if (include_x0) {
                noc_async_read_page(tile, x0_real, x0_real_ptr);
                noc_async_read_page(tile, x0_imag, x0_imag_ptr);
            }
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
            if (include_x0) {
                noc_async_read_page(tile, x0_real, x0_real_ptr);
                noc_async_read_barrier();
                noc_async_read_page(tile, x0_imag, x0_imag_ptr);
                noc_async_read_barrier();
            }
        }
    }
    if constexpr (batch_reads) {
        noc_async_read_barrier();
    }
    add_profile_cycles(
        counters.noc_read,
        counters.warmup_noc_read,
        get_timestamp_32b() - read_start,
        warmup);
    if constexpr (!fuse_s) {
        cb_push_back(cb_r_real, block_count);
    }
    cb_push_back(cb_r_negative_imag, block_count);
    cb_push_back(cb_r_imag, block_count);
    if constexpr (fuse_s) {
        cb_push_back(cb_r_negative_real, block_count);
    }
    if (include_x0) {
        cb_push_back(cb_x0_real, block_count);
        cb_push_back(cb_x0_imag, block_count);
    }
    constexpr std::uint32_t read_count = 3;
    const std::uint32_t total_read_count = read_count + (include_x0 ? 2 : 0);
    counters.event_count += total_read_count * block_count;
    if (warmup) {
        counters.warmup_event_count += total_read_count * block_count;
    }
}

template <bool fuse_s, bool batch_reads, typename RAccessor, typename X0Accessor>
void read_matrix_reload_profiled(
    std::uint32_t tile_id,
    std::uint32_t iterations,
    const RAccessor& r_real,
    const RAccessor& r_negative_imag,
    const RAccessor& r_imag,
    const X0Accessor& x0_real,
    const X0Accessor& x0_imag,
    const RAccessor& r_negative_real,
    ProfileCounters& counters,
    bool warmup_tile) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const bool first_iteration = iteration == 0;
        const bool warmup = warmup_tile && first_iteration;
        if constexpr (batch_reads) {
            read_matrix_profiled<fuse_s>(
                tile_id,
                r_real,
                r_negative_imag,
                r_imag,
                x0_real,
                x0_imag,
                r_negative_real,
                counters,
                warmup,
                first_iteration);
        } else {
            if constexpr (!fuse_s) {
                read_tile_profiled(cb_r_real, tile_id, r_real, counters, warmup);
            }
            read_tile_profiled(cb_r_negative_imag, tile_id, r_negative_imag, counters, warmup);
            read_tile_profiled(cb_r_imag, tile_id, r_imag, counters, warmup);
            if constexpr (fuse_s) {
                read_tile_profiled(cb_r_negative_real, tile_id, r_negative_real, counters, warmup);
            }
            if (first_iteration) {
                read_tile_profiled(cb_x0_real, tile_id, x0_real, counters, warmup);
                read_tile_profiled(cb_x0_imag, tile_id, x0_imag, counters, warmup);
            }
        }
    }
}

template <bool fuse_s, bool batch_reads, typename RAccessor, typename X0Accessor>
void read_matrix_block_reload_profiled(
    std::uint32_t tile_id,
    std::uint32_t block_count,
    std::uint32_t iterations,
    const RAccessor& r_real,
    const RAccessor& r_negative_imag,
    const RAccessor& r_imag,
    const X0Accessor& x0_real,
    const X0Accessor& x0_imag,
    const RAccessor& r_negative_real,
    ProfileCounters& counters,
    bool warmup_tile) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const bool first_iteration = iteration == 0;
        read_matrix_block_profiled<fuse_s, batch_reads>(
            tile_id,
            block_count,
            r_real,
            r_negative_imag,
            r_imag,
            x0_real,
            x0_imag,
            r_negative_real,
            counters,
            warmup_tile && first_iteration,
            first_iteration);
    }
}

void write_profile(ProfileCounters& counters) {
    cb_reserve_back(cb_profile_reader, 1);
    volatile tt_l1_ptr std::uint32_t* profile =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(get_write_ptr(cb_profile_reader));
    for (std::uint32_t index = 0; index < profile_words; ++index) {
        profile[index] = 0;
    }
    const std::uint32_t total = counters.total_end - counters.total_start;
    const std::uint32_t warmup_total = counters.warmup_end - counters.warmup_start;
    const std::uint32_t section_sum = counters.cb_wait + counters.noc_read;
    const std::uint32_t warmup_section_sum = counters.warmup_cb_wait + counters.warmup_noc_read;
    profile[profile_total_offset] = total;
    profile[profile_cb_wait_offset] = counters.cb_wait;
    profile[profile_noc_read_offset] = counters.noc_read;
    profile[profile_section_sum_offset] = section_sum;
    profile[profile_residual_offset] = total - section_sum;
    profile[profile_event_count_offset] = counters.event_count;
    profile[profile_warmup_base + profile_total_offset] = warmup_total;
    profile[profile_warmup_base + profile_cb_wait_offset] = counters.warmup_cb_wait;
    profile[profile_warmup_base + profile_noc_read_offset] = counters.warmup_noc_read;
    profile[profile_warmup_base + profile_section_sum_offset] = warmup_section_sum;
    profile[profile_warmup_base + profile_residual_offset] = warmup_total - warmup_section_sum;
    profile[profile_warmup_base + profile_warmup_event_count_offset] = counters.warmup_event_count;
    profile[profile_ready_offset] = profile_magic;
    profile[profile_warmup_ready_offset] = profile_magic;
    cb_push_back(cb_profile_reader, 1);
}
}  // namespace

void kernel_main() {
    // Runtime addresses follow the host tensor order: R variants, both X0
    // halves, then resident constants.  TensorAccessor preserves the selected
    // placement for each address, including DRAM-backed inputs.
    constexpr std::uint32_t iterations = get_compile_time_arg_val(0);
    constexpr bool fuse_s = get_compile_time_arg_val(1) != 0;
    constexpr bool batch_reads = get_compile_time_arg_val(2) != 0;
    constexpr std::uint32_t matrix_block = get_compile_time_arg_val(3);
    constexpr bool reload_r = get_compile_time_arg_val(4) != 0;
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
        // Fused S has no positive R-real runtime argument.
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

    // The five reader mode arguments precede every TensorAccessorArgs group:
    // iterations, fuse_s, batch_reads, matrix_block, and reload_r.
    constexpr std::uint32_t first_input_compile_arg = 5;
    constexpr auto first_input_args = TensorAccessorArgs<first_input_compile_arg>();
    constexpr auto r_negative_imag_args = TensorAccessorArgs<
        (fuse_s ? first_input_compile_arg : first_input_args.next_compile_time_args_offset())>();
    constexpr auto r_imag_args =
        TensorAccessorArgs<r_negative_imag_args.next_compile_time_args_offset()>();
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

    const auto r_negative_imag = TensorAccessor(r_negative_imag_args, r_negative_imag_address);
    const auto r_imag = TensorAccessor(r_imag_args, r_imag_address);
    const auto x0_real = TensorAccessor(x0_real_args, x0_real_address);
    const auto x0_imag = TensorAccessor(x0_imag_args, x0_imag_address);
    const auto identity = TensorAccessor(identity_args, identity_address);
    const auto zero = TensorAccessor(zero_args, zero_address);

    const bool measure_core = start_tile == 0;
    ProfileCounters nonmeasure_counters{};
    if (!measure_core) {
        read_tile(cb_identity, 0, identity);
        read_tile(cb_zero, 0, zero);
        if constexpr (fuse_s) {
            const auto r_negative_real =
                TensorAccessor(r_negative_real_args, r_negative_real_address);
            if constexpr (matrix_block == 1) {
                for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
                    if constexpr (reload_r) {
                        read_matrix_reload_profiled<fuse_s, batch_reads>(
                            start_tile + offset,
                            iterations,
                            r_negative_imag,
                            r_negative_imag,
                            r_imag,
                            x0_real,
                            x0_imag,
                            r_negative_real,
                            nonmeasure_counters,
                            false);
                    } else {
                        read_matrix<fuse_s, batch_reads>(
                            start_tile + offset,
                            r_negative_imag,
                            r_negative_imag,
                            r_imag,
                            x0_real,
                            x0_imag,
                            r_negative_real);
                    }
                }
            } else {
                for (std::uint32_t offset = 0; offset < tile_count; offset += matrix_block) {
                    const std::uint32_t block_count =
                        (tile_count - offset < matrix_block) ? (tile_count - offset) : matrix_block;
                    if constexpr (reload_r) {
                        read_matrix_block_reload_profiled<fuse_s, batch_reads>(
                            start_tile + offset,
                            block_count,
                            iterations,
                            r_negative_imag,
                            r_negative_imag,
                            r_imag,
                            x0_real,
                            x0_imag,
                            r_negative_real,
                            nonmeasure_counters,
                            false);
                    } else {
                        read_matrix_block<fuse_s, batch_reads>(
                            start_tile + offset,
                            block_count,
                            r_negative_imag,
                            r_negative_imag,
                            r_imag,
                            x0_real,
                            x0_imag,
                            r_negative_real);
                    }
                }
            }
        } else {
            const auto r_real = TensorAccessor(first_input_args, r_real_address);
            if constexpr (matrix_block == 1) {
                for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
                    if constexpr (reload_r) {
                        read_matrix_reload_profiled<fuse_s, batch_reads>(
                            start_tile + offset,
                            iterations,
                            r_real,
                            r_negative_imag,
                            r_imag,
                            x0_real,
                            x0_imag,
                            r_real,
                            nonmeasure_counters,
                            false);
                    } else {
                        read_matrix<fuse_s, batch_reads>(
                            start_tile + offset,
                            r_real,
                            r_negative_imag,
                            r_imag,
                            x0_real,
                            x0_imag,
                            r_real);
                    }
                }
            } else {
                for (std::uint32_t offset = 0; offset < tile_count; offset += matrix_block) {
                    const std::uint32_t block_count =
                        (tile_count - offset < matrix_block) ? (tile_count - offset) : matrix_block;
                    if constexpr (reload_r) {
                        read_matrix_block_reload_profiled<fuse_s, batch_reads>(
                            start_tile + offset,
                            block_count,
                            iterations,
                            r_real,
                            r_negative_imag,
                            r_imag,
                            x0_real,
                            x0_imag,
                            r_real,
                            nonmeasure_counters,
                            false);
                    } else {
                        read_matrix_block<fuse_s, batch_reads>(
                            start_tile + offset,
                            block_count,
                            r_real,
                            r_negative_imag,
                            r_imag,
                            x0_real,
                            x0_imag,
                            r_real);
                    }
                }
            }
        }
        return;
    }

    ProfileCounters counters;
    counters.total_start = get_timestamp_32b();
    counters.warmup_start = counters.total_start;
    read_tile_profiled(cb_identity, 0, identity, counters, true);
    read_tile_profiled(cb_zero, 0, zero, counters, true);
    if constexpr (fuse_s) {
        const auto r_negative_real =
            TensorAccessor(r_negative_real_args, r_negative_real_address);
        if constexpr (matrix_block == 1) {
            for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
                const bool warmup = offset == 0;
                if constexpr (reload_r) {
                    read_matrix_reload_profiled<fuse_s, batch_reads>(
                        start_tile + offset,
                        iterations,
                        r_negative_imag,
                        r_negative_imag,
                        r_imag,
                        x0_real,
                        x0_imag,
                        r_negative_real,
                        counters,
                        warmup);
                } else if constexpr (batch_reads) {
                    read_matrix_profiled<fuse_s>(
                        start_tile + offset,
                        r_negative_imag,
                        r_negative_imag,
                        r_imag,
                        x0_real,
                        x0_imag,
                        r_negative_real,
                        counters,
                        warmup);
                } else {
                    read_tile_profiled(cb_r_negative_imag, start_tile + offset, r_negative_imag, counters, warmup);
                    read_tile_profiled(cb_r_imag, start_tile + offset, r_imag, counters, warmup);
                    read_tile_profiled(cb_r_negative_real, start_tile + offset, r_negative_real, counters, warmup);
                    read_tile_profiled(cb_x0_real, start_tile + offset, x0_real, counters, warmup);
                    read_tile_profiled(cb_x0_imag, start_tile + offset, x0_imag, counters, warmup);
                }
                if (warmup) {
                    counters.warmup_end = get_timestamp_32b();
                }
            }
        } else {
            for (std::uint32_t offset = 0; offset < tile_count; offset += matrix_block) {
                const std::uint32_t block_count =
                    (tile_count - offset < matrix_block) ? (tile_count - offset) : matrix_block;
                const bool warmup = offset == 0;
                if constexpr (reload_r) {
                    read_matrix_block_reload_profiled<fuse_s, batch_reads>(
                        start_tile + offset,
                        block_count,
                        iterations,
                        r_negative_imag,
                        r_negative_imag,
                        r_imag,
                        x0_real,
                        x0_imag,
                        r_negative_real,
                        counters,
                        warmup);
                } else {
                    read_matrix_block_profiled<fuse_s, batch_reads>(
                        start_tile + offset,
                        block_count,
                        r_negative_imag,
                        r_negative_imag,
                        r_imag,
                        x0_real,
                        x0_imag,
                        r_negative_real,
                        counters,
                        warmup);
                }
                if (warmup) {
                    counters.warmup_end = get_timestamp_32b();
                }
            }
        }
    } else {
        const auto r_real = TensorAccessor(first_input_args, r_real_address);
        if constexpr (matrix_block == 1) {
            for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
                const bool warmup = offset == 0;
                if constexpr (reload_r) {
                    read_matrix_reload_profiled<fuse_s, batch_reads>(
                        start_tile + offset,
                        iterations,
                        r_real,
                        r_negative_imag,
                        r_imag,
                        x0_real,
                        x0_imag,
                        r_real,
                        counters,
                        warmup);
                } else if constexpr (batch_reads) {
                    read_matrix_profiled<fuse_s>(
                        start_tile + offset,
                        r_real,
                        r_negative_imag,
                        r_imag,
                        x0_real,
                        x0_imag,
                        r_real,
                        counters,
                        warmup);
                } else {
                    read_tile_profiled(cb_r_real, start_tile + offset, r_real, counters, warmup);
                    read_tile_profiled(cb_r_negative_imag, start_tile + offset, r_negative_imag, counters, warmup);
                    read_tile_profiled(cb_r_imag, start_tile + offset, r_imag, counters, warmup);
                    read_tile_profiled(cb_x0_real, start_tile + offset, x0_real, counters, warmup);
                    read_tile_profiled(cb_x0_imag, start_tile + offset, x0_imag, counters, warmup);
                }
                if (warmup) {
                    counters.warmup_end = get_timestamp_32b();
                }
            }
        } else {
            for (std::uint32_t offset = 0; offset < tile_count; offset += matrix_block) {
                const std::uint32_t block_count =
                    (tile_count - offset < matrix_block) ? (tile_count - offset) : matrix_block;
                const bool warmup = offset == 0;
                if constexpr (reload_r) {
                    read_matrix_block_reload_profiled<fuse_s, batch_reads>(
                        start_tile + offset,
                        block_count,
                        iterations,
                        r_real,
                        r_negative_imag,
                        r_imag,
                        x0_real,
                        x0_imag,
                        r_real,
                        counters,
                        warmup);
                } else {
                    read_matrix_block_profiled<fuse_s, batch_reads>(
                        start_tile + offset,
                        block_count,
                        r_real,
                        r_negative_imag,
                        r_imag,
                        x0_real,
                        x0_imag,
                        r_real,
                        counters,
                        warmup);
                }
                if (warmup) {
                    counters.warmup_end = get_timestamp_32b();
                }
            }
        }
    }
    counters.total_end = get_timestamp_32b();
    write_profile(counters);
}
