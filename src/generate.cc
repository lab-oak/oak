#include <search/util/softmax.h>
#include <train/battle/compressed-frame.h>
#include <util/argparse.h>
#include <util/policy.h>
#include <util/print.h>
#include <util/random.h>
#include <util/search.h>
#include <util/team-building.h>

#include <atomic>
#include <csignal>
#include <thread>

#include <fcntl.h>
#include <stdio.h>

struct ProgramArgs : public SelfPlayAgentArgs {
  std::optional<uint64_t> &seed = kwarg("seed", "Global program seed");
  std::optional<std::string> &working_dir = kwarg("dir", "Save directory");
  size_t &threads =
      kwarg("threads", "Number of parallel self-play games to run")
          .set_default(std::max(1u, std::thread::hardware_concurrency() - 1));
  size_t &max_samples =
      kwarg("max-samples",
            "Number of samples to generate before program termination")
          .set_default(1 << 30);
  size_t &buffer_size =
      kwarg("buffer-size", "Size of battle buffer (Mb) before write")
          .set_default(8);
  size_t &max_build_traj =
      kwarg("max-build-traj",
            "Size of build buffer (No. of traj's) before write")
          .set_default(1 << 10);
  size_t &print_interval =
      kwarg("print-interval", "Number of seconds between status prints")
          .set_default(30);
  bool &debug_print =
      kwarg("debug-print", "Enable verbose prints (debug build only)")
          .set_default(true);
  bool &keep_node = kwarg("keep-node", "Use the applicable child node for the "
                                       "next search instead of a new node.")
                        .set_default(true);

  double &fast_search_prob =
      kwarg("fast-search-prob",
            "Probability a search with only fast-search-time is used")
          .set_default(0);
  int &max_battle_length =
      kwarg("max-battle-length",
            "Battles exceeding this many updates are dropped")
          .set_default(-1);
  double &battle_skip_prob =
      kwarg("battle-skip-prob", "Only search on t1 for build trajectory data")
          .set_default(0);

  std::string &teams_path =
      kwarg("teams", "Path to teams file").set_default("");
};

// Stats for sample team matchup matrix
struct MatchupMatrix {

  struct Entry {
    std::atomic<size_t> n;
    std::atomic<size_t> v;
  };

  size_t n_teams;
  size_t n_entries;
  Entry *entries;

  void resize(const auto n) {
    n_teams = n;
    n_entries = n_teams * (n_teams - 1) / 2;
    entries = new Entry[n_entries]{};
  }

  ~MatchupMatrix() { delete[] entries; }

  size_t flat(auto i, auto j) {
    if (i < j) {
      return flat(j, i);
    }
    const auto index = i * (i - 1) / 2 + j;
    assert(index < n_entries);
    return index;
  }

  Entry &operator()(auto i, auto j) { return entries[flat(i, j)]; }

  void update(auto i, auto j, auto score) {
    if (i == j) {
      return;
    } else if (i < j) {
      return update(j, i, 2 - score);
    }
    auto &entry = (*this)(i, j);
    entry.n.fetch_add(1);
    entry.v.fetch_add(score);
  }
};

namespace RuntimeData {
bool terminated = false;
bool suspended = false;
std::string start_datetime = std::format(
    "{:%F-%T}",
    std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
// filenames
std::atomic<size_t> battle_buffer_counter{};
std::atomic<size_t> build_buffer_counter{};
// global sums
std::atomic<size_t> frame_counter{};
std::atomic<size_t> traj_counter{};
std::atomic<size_t> update_counter{};
std::atomic<size_t> update_with_node_counter{};
// teams
TeamBuilding::Provider provider;
MatchupMatrix matchup_matrix;
std::vector<size_t> battle_lengths;
std::atomic<size_t> thread_id{};
}; // namespace RuntimeData

// gengar mirrors going to turn 999
// not sharp detection, confuse ray can technically end the game. leech too but
// ghosts don't get it so idk, psywave also?
bool endless_battle_check(const auto &p1, const auto &p2) {
  using namespace PKMN::Data;
  const auto is_ghost = [](const auto &set) {
    return (set.species == PKMN::Data::Species::Gastly) ||
           (set.species == PKMN::Data::Species::Haunter) ||
           (set.species == PKMN::Data::Species::Gengar);
  };
  const auto cant_hit_ghosts = [](const auto &set) {
    return std::all_of(set.moves.begin(), set.moves.end(), [](const auto move) {
      const auto type = PKMN::Data::move_data(move).type;
      const auto bp = PKMN::Data::move_data(move).bp;

      return (type == PKMN::Data::Type::Normal) ||
             (type == PKMN::Data::Type::Fighting) || (bp == 0);
    });
  };
  // slow in theory but probably doesn't matter
  return std::all_of(
      p1.begin(), p1.end(), [is_ghost, cant_hit_ghosts, &p2](const auto &set1) {
        return std::all_of(
            p2.begin(), p2.end(),
            [is_ghost, cant_hit_ghosts, &set1](const auto &set2) {
              return is_ghost(set1) && cant_hit_ghosts(set2) &&
                     is_ghost(set2) && cant_hit_ghosts(set1);
            });
      });
}

void generate(const ProgramArgs *args_ptr) {
  const auto &args = *args_ptr;
  mt19937 device{args.seed.value()};
  const auto id = RuntimeData::thread_id.fetch_add(1) % args.threads;
  auto &battle_length = RuntimeData::battle_lengths[id];

  const size_t training_frames_target_size = args.buffer_size << 20;
  const size_t thread_frame_buffer_size = (args.buffer_size + 1) << 20;
  auto buffer = new char[thread_frame_buffer_size]{};
  size_t frame_buffer_write_index = 0;
  // These are generated slowly so a vector is fine
  std::vector<Train::Build::Trajectory> build_buffer{};

  const auto save_battle_buffer_to_disk = [&buffer, thread_frame_buffer_size,
                                           &frame_buffer_write_index, &args]() {
    if (frame_buffer_write_index == 0) {
      return;
    }
    const auto filename =
        std::to_string(RuntimeData::battle_buffer_counter.fetch_add(1)) +
        ".battle.data";
    const auto full_path =
        std::filesystem::path{args.working_dir.value()} / filename;
    int fd = open(full_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
      const auto write_result = write(fd, buffer, frame_buffer_write_index);
      close(fd);
    } else {
      std::cerr << "Failed to write buffer to " << full_path << std::endl;
    }

    std::memset(buffer, 0, thread_frame_buffer_size);
    frame_buffer_write_index = 0;
  };

  const auto save_build_buffer_to_disk = [&build_buffer, &args]() {
    if (build_buffer.size() == 0) {
      return;
    }
    const auto filename =
        std::to_string(RuntimeData::build_buffer_counter.fetch_add(1)) +
        ".build.data";
    const auto full_path =
        std::filesystem::path{args.working_dir.value()} / filename;

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
      debug_print("build buffer - bytes to write: " + std::to_string(bytes_to_write));
      debug_print("build buffer - bytes written: " + std::to_string(written));
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

  while (true) {

    auto [p1_build_traj, p1_team_index] =
        RuntimeData::provider.get_trajectory(device);
    auto [p2_build_traj, p2_team_index] =
        RuntimeData::provider.get_trajectory(device);

    const auto &p1_team = p1_build_traj.terminal;
    const auto &p2_team = p2_build_traj.terminal;
    if (endless_battle_check(p1_team, p2_team)) {
      std::cout << "EBC check failed. Continuing." << std::endl;
      continue;
    }
    const bool p1_built = (p1_build_traj.updates.size() > 0);
    const bool p2_built = (p2_build_traj.updates.size() > 0);
    float p1_matchup = .5;
    float p2_matchup = .5;

    bool skip_battle = false;
    if ((p1_built || p2_built) && device.uniform() < args.battle_skip_prob) {
      skip_battle = true;
    }

    auto battle = PKMN::battle(p1_team, p2_team, device.uniform_64());
    auto options = PKMN::options();
    const auto result = PKMN::update(battle, 0, 0, options);

    MCTS::BattleData battle_data{battle, PKMN::durations(), result};

    Train::Battle::CompressedFrames training_frames{battle_data.battle};

    auto agent = RuntimeSearch::Agent{.search_time = args.search_time,
                                      .bandit_name = args.bandit_name,
                                      .network_path = args.network_path,
                                      .discrete_network = args.use_discrete,
                                      .matrix_ucb_name = args.matrix_ucb_name};
    if (agent.uses_network()) {
      agent.initialize_network(battle_data.battle);
    }

    auto nodes = RuntimeSearch::Nodes{};

    auto policy_options =
        RuntimePolicy::Options{.mode = args.policy_mode,
                               .temp = args.policy_temp,
                               .min_prob = args.policy_min,
                               .nash_weight = args.policy_nash_weight};
    const auto rollout_policy_options = RuntimePolicy::Options{
        .mode = 'e', .temp = 1.0, .min_prob = args.policy_min};

    battle_length = 0;
    try {

      while (!pkmn_result_type(battle_data.result)) {

        if ((args.max_battle_length >= 1) &&
            (battle_length >= args.max_battle_length)) {
          throw std::runtime_error{"Max game length exceeded"};
        }

        while (RuntimeData::suspended) {
          sleep(1);
        }

        if (RuntimeData::terminated) {
          save_battle_buffer_to_disk();
          save_build_buffer_to_disk();
          delete[] buffer;
          return;
        }

        debug_print(PKMN::battle_data_to_string(battle_data.battle,
                                          battle_data.durations));

        const bool use_fast = device.uniform() < args.fast_search_prob;
        agent.search_time =
            ((battle_length == 0) && skip_battle)
                ? args.t1_search_time.value()
                : (use_fast ? args.fast_search_time.value() : args.search_time);
        policy_options.mode =
            use_fast ? args.fast_policy_mode : args.policy_mode;
        MCTS::Output output{};

        const bool policy_rollout_only =
            agent.uses_network() && (agent.search_time == "1");

        if (policy_rollout_only) {

          if (args.keep_node) {
            throw std::runtime_error{
                "--keep-node not allowed when using policy rollouts: there's a "
                "bug I don't understand just yet."};
            return;
          }

          std::array<float, 9> p1_logits{};
          std::array<float, 9> p2_logits{};
          output.m = pkmn_gen1_battle_choices(
              &battle_data.battle, PKMN_PLAYER_P1,
              pkmn_result_p1(battle_data.result), output.p1_choices.data(),
              PKMN_GEN1_MAX_CHOICES);
          output.n = pkmn_gen1_battle_choices(
              &battle_data.battle, PKMN_PLAYER_P2,
              pkmn_result_p2(battle_data.result), output.p2_choices.data(),
              PKMN_GEN1_MAX_CHOICES);
          agent.network.value().inference<false>(
              battle_data.battle,
              *pkmn_gen1_battle_options_chance_durations(&options), output.m,
              output.n, output.p1_choices.data(), output.p2_choices.data(),
              p1_logits.data(), p2_logits.data());
          softmax(output.p1_empirical.data(), p1_logits.data(), output.m);
          softmax(output.p2_empirical.data(), p2_logits.data(), output.n);

          if constexpr (debug) {
            const auto [p1_labels, p2_labels] =
                PKMN::choice_labels(battle_data.battle, battle_data.result);

            std::cout << "\nP1 policy:" << std::endl;
            for (auto i = 0; i < output.m; ++i) {
              std::cout << p1_labels[i] << ": " << output.p1_empirical[i]
                        << ' ';
            }
            std::cout << std::endl;
            std::cout << "P2 policy:" << std::endl;
            for (auto i = 0; i < output.n; ++i) {
              std::cout << p2_labels[i] << ": " << output.p2_empirical[i]
                        << ' ';
            }
            std::cout << std::endl;
          }
        } else {
          output = RuntimeSearch::run(battle_data, nodes, agent);
          if (battle_length == 0) {
            p1_matchup = output.empirical_value;
            p2_matchup = 1 - output.empirical_value;
            if (skip_battle) {
              break;
            }
          }
        }

        const auto p1_index = RuntimePolicy::process_and_sample(
            device, output.p1_empirical, output.p1_nash,
            policy_rollout_only ? rollout_policy_options : policy_options);
        const auto p2_index = RuntimePolicy::process_and_sample(
            device, output.p2_empirical, output.p2_nash,
            policy_rollout_only ? rollout_policy_options : policy_options);
        const auto p1_choice = output.p1_choices[p1_index];
        const auto p2_choice = output.p2_choices[p2_index];
        training_frames.updates.emplace_back(output, p1_choice, p2_choice);

        // update battle, durations, result (state info)
        battle_data.result =
            PKMN::update(battle_data.battle, p1_choice, p2_choice, options);
        battle_data.durations = PKMN::durations(options);

        // set nodes
        const auto &obs = *reinterpret_cast<const MCTS::Obs *>(
            pkmn_gen1_battle_options_chance_actions(&options));
        if (args.keep_node) {
          const bool node_kept = nodes.update(p1_index, p2_index, obs);
          RuntimeData::update_with_node_counter.fetch_add(node_kept);
          // TODO
          nodes.reset_stats();
        } else {
          nodes.reset();
        }
        RuntimeData::update_counter.fetch_add(1);

        ++battle_length;
        debug_print("update: " + std::to_string(battle_length));
      }
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      continue;
    }

    if (!skip_battle) {
      // battle
      training_frames.result = battle_data.result;

      const auto n_bytes_frames = training_frames.n_bytes();
      training_frames.write(buffer + frame_buffer_write_index);
      frame_buffer_write_index += n_bytes_frames;
      // data
      if (RuntimeData::frame_counter.fetch_add(training_frames.updates.size()) >
          args.max_samples) {
        RuntimeData::terminated = true;
      }
      if (frame_buffer_write_index >= training_frames_target_size) {
        save_battle_buffer_to_disk();
      }
      //
      p1_build_traj.score = PKMN::score(battle_data.result);
      p2_build_traj.score = 1 - PKMN::score(battle_data.result);
    }

    // build
    const bool used_matrix_teams =
        !p1_built && !p2_built && !RuntimeData::provider.rb;
    if (used_matrix_teams) {
      assert(p1_team_index >= 0);
      assert(p2_team_index >= 0);
      RuntimeData::matchup_matrix.update(p1_team_index, p2_team_index,
                                         PKMN::score(battle_data.result));
    }
    if (p1_built) {
      p1_build_traj.value = p1_matchup;
      build_buffer.push_back(p1_build_traj);
      RuntimeData::traj_counter.fetch_add(1);
    }
    if (p2_built) {
      p2_build_traj.value = p2_matchup;
      build_buffer.push_back(p2_build_traj);
      RuntimeData::traj_counter.fetch_add(1);
    }
    if (build_buffer.size() >= args.max_build_traj) {
      save_build_buffer_to_disk();
    }
  }
}

void print_thread_fn(const ProgramArgs *args_ptr) {
  const auto &args = *args_ptr;
  size_t frames_done = 0;
  size_t traj_done = 0;
  while (true) {
    for (int i = 0; i < args.print_interval; ++i) {
      if (RuntimeData::terminated) {
        return;
      }
      sleep(1);
    }
    const auto frames_more = RuntimeData::frame_counter.load();
    const auto traj_more = RuntimeData::traj_counter.load();
    std::cout << (frames_more - frames_done) / (float)args.print_interval
              << " battle frames/sec." << std::endl;
    if (RuntimeData::provider.team_modify_prob > 0) {
      std::cout << (traj_more - traj_done) / (float)args.print_interval
                << " build traj./sec." << std::endl;
    }
    if (args.keep_node) {
      double keep_node_ratio =
          (double)RuntimeData::update_with_node_counter.load() /
          (double)RuntimeData::update_counter.load();
      std::cout << "keep node ratio: " << keep_node_ratio << std::endl;
    }
    const auto progress = (double)frames_more / args.max_samples * 100;
    std::cout << "progress: " << progress << "%" << std::endl;

    frames_done = frames_more;
    traj_done = traj_more;

    std::cout << "Game Lengths: ";
    for (const auto len : RuntimeData::battle_lengths) {
      std::cout << len << ' ';
    }
    std::cout << std::endl;
  }
}

void setup(const auto &args) {
  // optional args
  if (!args.seed.has_value()) {
    args.seed.emplace(std::random_device{}());
  }
  if (!args.working_dir.has_value()) {
    args.working_dir.emplace(RuntimeData::start_datetime);
  }
  if (!args.t1_search_time.has_value()) {
    args.t1_search_time.emplace(args.search_time);
  }
  if (!args.fast_search_time.has_value()) {
    args.fast_search_time.emplace(args.search_time);
  }

  // create working dir
  const std::filesystem::path working_dir = args.working_dir.value();
  std::error_code ec;
  const bool created = std::filesystem::create_directory(working_dir, ec);
  if (ec) {
    std::cerr << "Error creating directory: " << ec.message() << '\n';
    throw std::runtime_error("Could not create working dir.");
  } else if (created) {
    std::cout << "Created directory " << working_dir.string() << std::endl;
  } else {
    throw std::runtime_error("Could not create working dir.");
  }

  // save args
  {
    const auto args_path = working_dir / "args";
    std::ofstream args_file(args_path);
    if (!args_file) {
      throw std::runtime_error("Failed to open args.txt for writing.");
    }
    args.print(args_file);
  }

  // teams
  RuntimeData::provider = TeamBuilding::Provider{args.teams_path};
  RuntimeData::provider.omitter = {args.max_pokemon, args.pokemon_delete_prob,
                                   args.move_delete_prob};
  RuntimeData::provider.network_path = args.build_network_path;
  RuntimeData::provider.team_modify_prob = args.team_modify_prob;
  RuntimeData::provider.read_network_parameters();
  RuntimeData::matchup_matrix.resize(RuntimeData::provider.teams.size());

  // stats
  RuntimeData::battle_lengths.resize(args.threads);
}

void cleanup(const auto &args) {
  const std::filesystem::path working_dir = args.working_dir.value();
  const auto matchup_matrix_path = working_dir / "matchup-matrix";
  std::ofstream matchup_matrix_file(matchup_matrix_path);
  if (!matchup_matrix_file) {
    std::cerr << "Failed to open matchup matrix file" << std::endl;
  }
  for (auto i = 0; i < RuntimeData::matchup_matrix.n_teams; ++i) {
    for (auto j = 0; j < i; ++j) {
      matchup_matrix_file << i << ' ' << j << ": ";
      const auto &entry = RuntimeData::matchup_matrix(i, j);
      if (entry.n) {
        matchup_matrix_file << entry.v / 2.0 / entry.n << std::endl;
      } else {
        matchup_matrix_file << "N/A" << std::endl;
      }
    }
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

int main(int argc, char **argv) {
  std::signal(SIGINT, handle_terminate);
  std::signal(SIGTSTP, handle_suspend);

  auto args = argparse::parse<ProgramArgs>(argc, argv);

  setup(args);

  mt19937 device{args.seed.value()};

  std::vector<std::thread> thread_pool;
  for (int t = 0; t < args.threads; ++t) {
    thread_pool.emplace_back(generate, &args);
  }
  std::thread print_thread{print_thread_fn, &args};

  for (auto &th : thread_pool) {
    th.join();
  }
  print_thread.join();

  cleanup(args);

  return 0;
}
