// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <nlohmann/json.hpp>
#include <tt-metalium/mesh_device.hpp>

nlohmann::json run_frame_coverage_probe(
    tt::tt_metal::distributed::MeshDevice& mesh,
    uint32_t repeat_count, bool watcher, uint32_t cap_seconds);
