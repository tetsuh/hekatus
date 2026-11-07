// SPDX-License-Identifier: Apache-2.0
// CPU-only result publication; no Metalium or device dependencies.
#pragma once

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace issue120 {
inline bool valid_attempt_identity(const std::string& identity) {
    const auto alnum = [](unsigned char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
    };
    return !identity.empty() && identity.size() <= 128 && alnum(identity.front()) &&
        std::all_of(identity.begin(), identity.end(), [&](unsigned char c) {
            return alnum(c) || c == '_' || c == '-' || c == '.';
        });
}

inline void validate_result_target(
    const std::filesystem::path& root, const std::filesystem::path& target,
    const std::string& identity) {
    if (!valid_attempt_identity(identity)) {
        throw std::invalid_argument("invalid wrapper attempt identity");
    }
    const auto expected = root / ("native-" + identity) / "result" / "result.json";
    if (!root.is_absolute() || target != expected || std::filesystem::canonical(root) != root ||
        std::filesystem::canonical(target.parent_path()) != target.parent_path() ||
        !std::filesystem::is_empty(target.parent_path())) {
        throw std::invalid_argument("native result target must be a new program-owned slot");
    }
}

inline void write_all(int fd, const std::string& bytes) {
    size_t offset = 0;
    while (offset < bytes.size()) {
        const auto written = ::write(fd, bytes.data() + offset, bytes.size() - offset);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            throw std::runtime_error("native result write failed");
        }
        offset += static_cast<size_t>(written);
    }
}

inline void publish_result(
    const std::filesystem::path& root, const std::filesystem::path& target,
    const std::string& identity, const std::string& payload) {
    validate_result_target(root, target, identity);
    const std::string bytes = "{\"schema\":\"issue120-native-result-v1\",\"attempt_identity\":\"" +
        identity + "\",\"result\":" + payload + "}\n";
    const std::string temporary = target.string() + ".tmp";
    int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) {
        throw std::runtime_error("native result temporary file cannot be claimed");
    }
    try {
        write_all(fd, bytes);
        if (::fsync(fd) != 0) {
            throw std::runtime_error("native result fsync failed");
        }
        const int closed = ::close(fd);
        fd = -1;
        if (closed != 0) {
            throw std::runtime_error("native result close failed");
        }
        // Linux pinned-image contract: atomic publication without replacing data.
        if (::syscall(SYS_renameat2, AT_FDCWD, temporary.c_str(), AT_FDCWD,
                      target.c_str(), RENAME_NOREPLACE) != 0) {
            throw std::runtime_error("native result atomic rename failed");
        }
        const int directory_fd = ::open(target.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory_fd < 0) {
            throw std::runtime_error("native result directory cannot be synced");
        }
        const int synced = ::fsync(directory_fd);
        const int directory_closed = ::close(directory_fd);
        if (synced != 0 || directory_closed != 0) {
            throw std::runtime_error("native result directory sync failed");
        }
    } catch (...) {
        if (fd >= 0) {
            ::close(fd);
        }
        // Preserve partial temporary bytes on failure; never publish them as a result.
        throw;
    }
}
}  // namespace issue120
