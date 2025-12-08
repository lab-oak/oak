#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

struct BattleFrameBuffer {

  std::vector<char> buffer;
  size_t write_index;

  BattleFrameBuffer(size_t size) : buffer{}, write_index{} {
    buffer.resize(size);
  }

  void clear() {
    std::fill(buffer.begin(), buffer.end(), 0);
    write_index = 0;
  }

  void save_to_disk(std::filesystem::path dir, auto &atomic) {
    if (write_index == 0) {
      return;
    }
    const auto filename = std::to_string(atomic.fetch_add(1)) + ".battle.data";
    const auto full_path = dir / filename;
    int fd = open(full_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
      const auto write_result = write(fd, buffer.data(), write_index);
      close(fd);
    } else {
      std::cerr << "Failed to write buffer to " << full_path << std::endl;
    }

    clear();
  }

  void write_frames(const auto &training_frames) {
    const auto n_bytes_frames = training_frames.n_bytes();
    training_frames.write(buffer.data() + write_index);
    write_index += n_bytes_frames;
  }
};
