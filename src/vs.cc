#include <format/random-battles/randbat.h>
#include <libpkmn/data.h>
#include <libpkmn/strings.h>
#include <nn/battle/network.h>
#include <teams/ou-sample-teams.h>
#include <train/battle/compressed-frame.h>
#include <util/policy.h>
#include <util/random.h>
#include <util/search.h>
#include <util/team-building.h>

#include <atomic>
#include <cmath>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

auto inverse_sigmoid(const auto x) { return std::log(x) - std::log(1 - x); }

void print(const auto &data, const bool newline = true) {
  std::cout << data;
  if (newline) {
    std::cout << '\n';
  }
}

namespace RuntimeOptions {
size_t threads = 0;
double print_prob = 0;
double early_stop_log = 3.5;

RuntimeSearch::Agent p1_agent{"4096", "exp3-0.03", "mc"};
RuntimeSearch::Agent p2_agent{"4096", "exp3-0.03", "mc"};

RuntimePolicy::Options p1_policy_options{};
RuntimePolicy::Options p2_policy_options{};

size_t buffer_size_mb = 8;
size_t max_build_traj = 1 << 10;

namespace TeamGen {
// std::string teams_path = "";
std::string network_path = "";
NN::Build::Network network{};

double team_modify_prob = 0;
TeamBuilding::Omitter omitter{};
double battle_skip_prob = 0;
}; // namespace TeamGen

} // namespace RuntimeOptions

auto generate_team(mt19937 &device, const auto &base_team)
    -> Train::Build::Trajectory {
  using namespace RuntimeOptions::TeamGen;

  const auto base_team_vec =
      std::vector<PKMN::Set>(base_team.begin(), base_team.end());

  auto team = base_team_vec;
  const bool changed = omitter.shuffle_and_truncate(device, team);
  bool deleted = false;
  if (device.uniform() < team_modify_prob) {
    deleted = omitter.delete_info(device, team);
  }

  if (!deleted) {
    Train::Build::Trajectory trajectory{};
    trajectory.initial = trajectory.terminal = team;
    return trajectory;
  } else {
    const auto trajectory =
        TeamBuilding::rollout_build_network(device, network, team);
    assert(trajectory.updates.size() > 0);
    return trajectory;
  }
}

bool parse_options(int argc, char **argv) {
  using namespace RuntimeOptions;

  if (argc < 2) {
    std::cout << "Usage: ./vs [OPTIONS]\nArg '--threads=' is "
                 "required.\n--help for more."
              << std::endl;
    return true;
  }

  std::vector<char *> args(argv + 1, argv + argc);
  assert(args.size() == argc - 1);

  for (auto &a : args) {
    if (a == nullptr) {
      continue;
    }
    char *b = nullptr;
    std::swap(a, b);
    std::string arg{b};
    if (arg.starts_with("--help")) {
      std::cout << "TODO help text" << std::endl;
      return true;
    } else if (arg.starts_with("--threads=")) {
      threads = std::stoul(arg.substr(10));
    } else if (arg.starts_with("--p1-search-time=")) {
      p1_agent.search_time = arg.substr(17);
    } else if (arg.starts_with("--p2-search-time=")) {
      p2_agent.search_time = arg.substr(17);
    } else if (arg.starts_with("--p1-bandit-name=")) {
      p1_agent.bandit_name = arg.substr(17);
    } else if (arg.starts_with("--p2-bandit-name=")) {
      p2_agent.bandit_name = arg.substr(17);
    } else if (arg.starts_with("--p1-network-path=")) {
      p1_agent.network_path = arg.substr(18);
    } else if (arg.starts_with("--p2-network-path=")) {
      p2_agent.network_path = arg.substr(18);
    } else if (arg.starts_with("--p1-policy-mode=")) {
      p1_policy_options.mode = arg[17];
    } else if (arg.starts_with("--p2-policy-mode=")) {
      p2_policy_options.mode = arg[17];

    } else if (arg.starts_with("--max-pokemon=")) {
      TeamGen::omitter.max_pokemon = std::stoul(arg.substr(14));
    } else if (arg.starts_with("--build-network-path=")) {
      TeamGen::network_path = arg.substr(21);
    } else if (arg.starts_with("--team-modify-prob=")) {
      TeamGen::team_modify_prob = std::stod(arg.substr(19));
    } else if (arg.starts_with("--pokemon-delete-prob=")) {
      TeamGen::omitter.pokemon_delete_prob = std::stod(arg.substr(22));
    } else if (arg.starts_with("--move-delete-prob=")) {
      TeamGen::omitter.move_delete_prob = std::stod(arg.substr(19));

    } else if (arg.starts_with("--print-prob=")) {
      print_prob = std::stof(arg.substr(13));
    } else if (arg.starts_with("--early-stop-log=")) {
      early_stop_log = std::stof(arg.substr(17));
    } else {
      std::swap(a, b);
    }
  }

  for (auto a : args) {
    if (a != nullptr) {
      throw std::runtime_error{"Unrecognized arg: " + std::string(a)};
    }
  }

  return threads == 0;
}

namespace RuntimeData {
bool terminated = false;
bool suspended = false;

std::atomic<size_t> score{};
std::atomic<size_t> n{};

std::filesystem::path working_dir = std::format(
    "vs-{:%F-%T}",
    std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
// filenames
std::atomic<size_t> battle_buffer_counter{};
std::atomic<size_t> build_buffer_counter{};
} // namespace RuntimeData

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

  void save_to_disk(std::filesystem::path dir) {
    if (write_index == 0) {
      return;
    }
    const auto filename =
        std::to_string(RuntimeData::battle_buffer_counter.fetch_add(1)) +
        ".battle";
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

void thread_fn(uint64_t seed) {
  mt19937 device{seed};

  // data gen
  const size_t training_frames_target_size = RuntimeOptions::buffer_size_mb
                                             << 20;
  const size_t thread_frame_buffer_size = (RuntimeOptions::buffer_size_mb + 1)
                                          << 20;

  BattleFrameBuffer p1_battle_frame_buffer{thread_frame_buffer_size};
  BattleFrameBuffer p2_battle_frame_buffer{thread_frame_buffer_size};

  // These are generated slowly so a vector is fine
  std::vector<Train::Build::Trajectory> build_buffer{};

  const auto save_build_buffer_to_disk = [&build_buffer]() {
    if (build_buffer.size() == 0) {
      return;
    }
    const auto filename =
        std::to_string(RuntimeData::build_buffer_counter.fetch_add(1)) +
        ".build";
    const auto full_path = RuntimeData::working_dir / filename;

    const int fd = open(full_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
      const size_t bytes_to_write =
          build_buffer.size() *
          (Encode::Build::CompressedTrajectory<>::size_no_team);
      ssize_t written = 0;
      for (const auto &trajectory : build_buffer) {
        const auto traj = Encode::Build::CompressedTrajectory<>{trajectory};
        written += write(fd, &traj, decltype(traj)::size_no_team);
      }
      close(fd);
      if (written != static_cast<ssize_t>(bytes_to_write)) {
        std::cerr << "Short write when flushing build buffer to " << full_path
                  << " (" << written << '/' << bytes_to_write << " bytes)\n";
      }
    } else {
      std::cerr << "Failed to open " << full_path << " for writing\n";
    }

    build_buffer.clear();
  };

  const auto play = [&](auto p1_build_traj, auto p2_build_traj) -> int {
    int score_2;

    const auto &p1_team = p1_build_traj.terminal;
    const auto &p2_team = p2_build_traj.terminal;

    auto battle = PKMN::battle(p1_team, p2_team, device.uniform_64());
    auto options = PKMN::options();
    const auto result = PKMN::update(battle, 0, 0, options);
    MCTS::BattleData battle_data{battle, PKMN::durations(), result};

    auto p1_agent_local = RuntimeOptions::p1_agent;
    auto p2_agent_local = RuntimeOptions::p2_agent;
    if (p1_agent_local.uses_network()) {
      p1_agent_local.read_network_parameters();
      p1_agent_local.network.value().fill_cache(battle);
    }
    if (p2_agent_local.uses_network()) {
      p2_agent_local.read_network_parameters();
      p2_agent_local.network.value().fill_cache(battle);
    }

    Train::Battle::CompressedFrames p1_battle_frames{battle};
    Train::Battle::CompressedFrames p2_battle_frames{battle};

    int p1_early_stop = 0;
    int p2_early_stop = 0;
    bool early_stop = false;
    size_t updates = 0;
    try {
      // playout game
      while (!pkmn_result_type(battle_data.result)) {

        while (RuntimeData::suspended) {
          sleep(1);
        }

        if (RuntimeData::terminated) {
          using Ignored = int;
          return Ignored{};
        }

        const auto [p1_choices, p2_choices] =
            PKMN::choices(battle_data.battle, battle_data.result);

        MCTS::Output p1_output, p2_output;
        int p1_index = 0;
        int p2_index = 0;
        p1_early_stop = 0;
        p2_early_stop = 0;
        // TODO - make up your mind how this is going to be handled
        // if (p1_choices.size() > 1) {
        {
          RuntimeSearch::Nodes nodes{};
          p1_output = RuntimeSearch::run(battle_data, nodes, p1_agent_local);
          p1_early_stop = inverse_sigmoid(p1_output.empirical_value) /
                          RuntimeOptions::early_stop_log;
          p1_index = process_and_sample(device, p1_output.p1_empirical,
                                        p1_output.p1_nash,
                                        RuntimeOptions::p1_policy_options);
        }
        // if (p2_choices.size() > 1) {
        {
          RuntimeSearch::Nodes nodes{};
          p2_output = RuntimeSearch::run(battle_data, nodes, p2_agent_local);
          p2_early_stop = inverse_sigmoid(p2_output.empirical_value) /
                          RuntimeOptions::early_stop_log;
          p2_index = process_and_sample(device, p2_output.p2_empirical,
                                        p2_output.p2_nash,
                                        RuntimeOptions::p2_policy_options);
        }

        if (updates == 0) {
          p1_build_traj.value = p1_output.empirical_value;
          p2_build_traj.value = 1 - p2_output.empirical_value;
        }

        // only if they have same sign and are both non zero
        if ((p1_early_stop * p2_early_stop) > 0) {
          early_stop = true;
          break;
        }

        const auto p1_choice = p1_choices[p1_index];
        const auto p2_choice = p2_choices[p2_index];

        p1_battle_frames.updates.emplace_back(p1_output, p1_choice, p2_choice);
        p2_battle_frames.updates.emplace_back(p2_output, p1_choice, p2_choice);

        if (device.uniform() < RuntimeOptions::print_prob) {
          print("GAME: " + std::to_string(RuntimeData::n.load()), false);
          print(" UPDATE: " + std::to_string(updates));
          print(PKMN::battle_data_to_string(battle_data.battle,
                                            battle_data.durations));
        }
        battle_data.result =
            PKMN::update(battle_data.battle, p1_choice, p2_choice, options);
        battle_data.durations = PKMN::durations(options);
        ++updates;
      }

      if (early_stop) {
        if (p1_early_stop > 0) {
          score_2 = 2;
          battle_data.result = PKMN::result(PKMN::Result::Win);
        } else {
          score_2 = 0;
          battle_data.result = PKMN::result(PKMN::Result::Lose);
        }
      } else {
        score_2 = PKMN::score2(battle_data.result);
      }
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
    }

    p1_battle_frames.result = battle_data.result;
    p2_battle_frames.result = battle_data.result;

    p1_battle_frame_buffer.write_frames(p1_battle_frames);
    p2_battle_frame_buffer.write_frames(p2_battle_frames);

    return score_2;
  };

  const auto randbat_traj = [&]() {
    Train::Build::Trajectory trajectory{};
    const auto seed = std::bit_cast<int64_t>(device.uniform_64());
    RandomBattles::PRNG prng{seed};
    RandomBattles::Teams t{prng};
    // completed but in weird format
    const auto partial_team = t.randomTeam();
    const auto team = t.partialToTeam(partial_team);
    std::vector<PKMN::Set> team_vec(team.begin(), team.end());
    trajectory.initial = team_vec;
    trajectory.terminal = team_vec;
    return trajectory;
  };

  while (true) {
    const auto p1_base_team = Teams::ou_sample_teams[device.random_int(
        Teams::ou_sample_teams.size())];
    const auto p2_base_team = Teams::ou_sample_teams[device.random_int(
        Teams::ou_sample_teams.size())];

    // const auto p1_build_traj = generate_team(device, p1_base_team);
    // const auto p2_build_traj = generate_team(device, p2_base_team);

    const auto p1_build_traj = randbat_traj();
    const auto p2_build_traj = randbat_traj();

    const auto s1 = play(p1_build_traj, p2_build_traj);
    const auto s2 = play(p2_build_traj, p1_build_traj);

    if (!RuntimeData::terminated) {
      RuntimeData::score.fetch_add(s1 + s2);
      RuntimeData::n.fetch_add(2);
    }

    if (p1_battle_frame_buffer.write_index >= training_frames_target_size) {
      p1_battle_frame_buffer.save_to_disk(RuntimeData::working_dir / "p1");
    }
    if (p2_battle_frame_buffer.write_index >= training_frames_target_size) {
      p2_battle_frame_buffer.save_to_disk(RuntimeData::working_dir / "p2");
    }

    if (RuntimeData::terminated) {
      p1_battle_frame_buffer.save_to_disk(RuntimeData::working_dir / "p1");
      p2_battle_frame_buffer.save_to_disk(RuntimeData::working_dir / "p2");
      save_build_buffer_to_disk();
      return;
    }

    if (build_buffer.size() >= RuntimeOptions::max_build_traj) {
      save_build_buffer_to_disk();
    }
  }

  return;
}

void progress_thread_fn(int sec) {
  while (true) {
    for (int s = 0; s < sec; ++s) {
      if (RuntimeData::terminated) {
        return;
      }
      sleep(1);
    }
    std::cout << "score: "
              << (RuntimeData::score.load() / 2.0 / RuntimeData::n.load())
              << " over " << RuntimeData::n.load() << " games." << std::endl;
  }
}

void handle_suspend(int signal) {
  RuntimeData::suspended = !RuntimeData::suspended;
  std::cout << (RuntimeData::suspended ? "Suspended." : "Resumed.")
            << std::endl;
}

void handle_terminate(int signal) {
  RuntimeData::terminated = true;
  RuntimeData::suspended = false;
  std::cout << "Terminated." << std::endl;
}

void setup(auto &device) {

  // create working dir
  std::error_code ec;
  bool created =
      std::filesystem::create_directory(RuntimeData::working_dir, ec) &&
      std::filesystem::create_directory(RuntimeData::working_dir / "p1", ec) &&
      std::filesystem::create_directory(RuntimeData::working_dir / "p2", ec);
  if (ec) {
    std::cerr << "Error creating directory: " << ec.message() << '\n';
    throw std::runtime_error("Could not create working dir.");
  } else if (created) {
    std::cout << "Created directory " << RuntimeData::working_dir.string()
              << std::endl;
  } else {
    throw std::runtime_error("Could not create working dir.");
  }

  // save args
  // TODO

  // global build network
  using namespace RuntimeOptions::TeamGen;
  if (!network_path.empty()) {
    std::ifstream file{RuntimeOptions::TeamGen::network_path};
    network.read_parameters(file);
  } else {
    network.initialize(device);
  }
}

int main(int argc, char **argv) {

  std::signal(SIGINT, handle_terminate);
  std::signal(SIGTSTP, handle_suspend);

  if (const bool exit_early = parse_options(argc, argv)) {
    return 1;
  }

  size_t seed = std::random_device{}();

  mt19937 device{seed};

  setup(device);

  std::vector<std::thread> thread_pool{};
  for (auto t = 0; t < RuntimeOptions::threads; ++t) {
    thread_pool.emplace_back(std::thread{&thread_fn, device.uniform_64()});
  }
  auto progress_thread = std::thread(&progress_thread_fn, 30);
  for (auto &thread : thread_pool) {
    thread.join();
  }
  progress_thread.join();

  std::cout << "score: "
            << (RuntimeData::score.load() / 2.0 / RuntimeData::n.load())
            << " over " << RuntimeData::n.load() << " games." << std::endl;
}