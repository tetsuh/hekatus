// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr std::uint32_t cb_output_real = 15;
constexpr std::uint32_t cb_output_imag = 16;
constexpr std::uint32_t cb_profile_reader = 17;
constexpr std::uint32_t cb_profile_compute = 18;
constexpr std::uint32_t cb_profile_writer = 19;
constexpr std::uint32_t profile_magic = 0x5052464C;
constexpr std::uint32_t profile_total_offset = 0;
constexpr std::uint32_t profile_cb_wait_offset = 1;
constexpr std::uint32_t profile_noc_write_offset = 2;
constexpr std::uint32_t profile_section_sum_offset = 8;
constexpr std::uint32_t profile_residual_offset = 9;
constexpr std::uint32_t profile_event_count_offset = 10;
constexpr std::uint32_t profile_warmup_event_count_offset = 11;
constexpr std::uint32_t profile_warmup_base = 32;
constexpr std::uint32_t profile_ready_offset = 31;
constexpr std::uint32_t profile_warmup_ready_offset = 63;
constexpr std::uint32_t profile_slot_stride = 64;
constexpr std::uint32_t profile_words = 32 * 32;

struct ProfileCounters {
    std::uint32_t total_start = 0;
    std::uint32_t total_end = 0;
    std::uint32_t warmup_start = 0;
    std::uint32_t warmup_end = 0;
    std::uint32_t event_count = 0;
    std::uint32_t warmup_event_count = 0;
    std::uint32_t cb_wait = 0;
    std::uint32_t noc_write = 0;
    std::uint32_t warmup_cb_wait = 0;
    std::uint32_t warmup_noc_write = 0;
};

void add_profile_cycles(std::uint32_t& total, std::uint32_t& warmup, std::uint32_t cycles, bool is_warmup) {
    total += cycles;
    if (is_warmup) {
        warmup += cycles;
    }
}

void fill_profile_writer(ProfileCounters& counters) {
    cb_reserve_back(cb_profile_writer, 1);
    volatile tt_l1_ptr std::uint32_t* profile =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(get_write_ptr(cb_profile_writer));
    for (std::uint32_t index = 0; index < profile_words; ++index) {
        profile[index] = 0;
    }
    const std::uint32_t total = counters.total_end - counters.total_start;
    const std::uint32_t warmup_total = counters.warmup_end - counters.warmup_start;
    const std::uint32_t section_sum = counters.cb_wait + counters.noc_write;
    const std::uint32_t warmup_section_sum = counters.warmup_cb_wait + counters.warmup_noc_write;
    profile[profile_total_offset] = total;
    profile[profile_cb_wait_offset] = counters.cb_wait;
    profile[profile_noc_write_offset] = counters.noc_write;
    profile[profile_section_sum_offset] = section_sum;
    profile[profile_residual_offset] = total - section_sum;
    profile[profile_event_count_offset] = counters.event_count;
    profile[profile_warmup_base + profile_total_offset] = warmup_total;
    profile[profile_warmup_base + profile_cb_wait_offset] = counters.warmup_cb_wait;
    profile[profile_warmup_base + profile_noc_write_offset] = counters.warmup_noc_write;
    profile[profile_warmup_base + profile_section_sum_offset] = warmup_section_sum;
    profile[profile_warmup_base + profile_residual_offset] = warmup_total - warmup_section_sum;
    profile[profile_warmup_base + profile_warmup_event_count_offset] = counters.warmup_event_count;
    profile[profile_ready_offset] = profile_magic;
    profile[profile_warmup_ready_offset] = profile_magic;
    cb_push_back(cb_profile_writer, 1);
}
}  // namespace

void kernel_main() {
    const std::uint32_t real_address = get_arg_val<std::uint32_t>(0);
    const std::uint32_t imag_address = get_arg_val<std::uint32_t>(1);
    const std::uint32_t profile_address = get_arg_val<std::uint32_t>(2);
    const std::uint32_t profile_index = get_arg_val<std::uint32_t>(3);
    const std::uint32_t start_tile = get_arg_val<std::uint32_t>(4);
    const std::uint32_t tile_count = get_arg_val<std::uint32_t>(5);

    constexpr auto real_args = TensorAccessorArgs<0>();
    constexpr auto imag_args = TensorAccessorArgs<real_args.next_compile_time_args_offset()>();
    constexpr auto profile_args = TensorAccessorArgs<imag_args.next_compile_time_args_offset()>();
    const auto real = TensorAccessor(real_args, real_address);
    const auto imag = TensorAccessor(imag_args, imag_address);
    const auto profile = TensorAccessor(profile_args, profile_address);
    const bool measure_core = start_tile == 0;

    ProfileCounters counters;
    if (measure_core) {
        counters.total_start = get_timestamp_32b();
        counters.warmup_start = counters.total_start;
    }
    for (std::uint32_t offset = 0; offset < tile_count; ++offset) {
        const std::uint32_t tile = start_tile + offset;
        const bool warmup = measure_core && offset == 0;
        const std::uint32_t wait_start = measure_core ? get_timestamp_32b() : 0;
        cb_wait_front(cb_output_real, 1);
        cb_wait_front(cb_output_imag, 1);
        if (measure_core) {
            add_profile_cycles(
                counters.cb_wait,
                counters.warmup_cb_wait,
                get_timestamp_32b() - wait_start,
                warmup);
        }

        const std::uint32_t write_start = measure_core ? get_timestamp_32b() : 0;
        noc_async_write_page(tile, real, get_read_ptr(cb_output_real));
        noc_async_write_barrier();
        noc_async_write_page(tile, imag, get_read_ptr(cb_output_imag));
        noc_async_write_barrier();
        if (measure_core) {
            add_profile_cycles(
                counters.noc_write,
                counters.warmup_noc_write,
                get_timestamp_32b() - write_start,
                warmup);
            ++counters.event_count;
            if (warmup) {
                ++counters.warmup_event_count;
            }
        }
        cb_pop_front(cb_output_real, 1);
        cb_pop_front(cb_output_imag, 1);
        if (measure_core && warmup) {
            counters.warmup_end = get_timestamp_32b();
        }
    }
    if (!measure_core) {
        return;
    }
    counters.total_end = get_timestamp_32b();
    cb_wait_front(cb_profile_reader, 1);
    cb_wait_front(cb_profile_compute, 1);
    fill_profile_writer(counters);
    cb_wait_front(cb_profile_writer, 1);
    noc_async_write_page(profile_index, profile, get_read_ptr(cb_profile_reader));
    noc_async_write_barrier();
    noc_async_write_page(profile_index + 1, profile, get_read_ptr(cb_profile_compute));
    noc_async_write_barrier();
    noc_async_write_page(profile_index + 2, profile, get_read_ptr(cb_profile_writer));
    noc_async_write_barrier();
    // Clear completion flags after the DRAM copy so the next launch cannot
    // mistake the previous triplet for a newly completed compute page.
    volatile tt_l1_ptr std::uint32_t* compute_profile = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_read_ptr(cb_profile_compute));
    compute_profile[profile_ready_offset] = 0;
    compute_profile[profile_warmup_ready_offset] = 0;
    compute_profile[profile_slot_stride + profile_ready_offset] = 0;
    compute_profile[profile_slot_stride + profile_warmup_ready_offset] = 0;
    compute_profile[2 * profile_slot_stride + profile_ready_offset] = 0;
    compute_profile[2 * profile_slot_stride + profile_warmup_ready_offset] = 0;
    cb_pop_front(cb_profile_reader, 1);
    cb_pop_front(cb_profile_compute, 1);
    cb_pop_front(cb_profile_writer, 1);
}
