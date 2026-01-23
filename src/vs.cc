#include <train/battle/compressed-frame.h>
#include <util/argparse.h>
#include <util/battle-frame-buffer.h>
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

struct ProgramArgs : public VsAgentArgs {
  std::optional<uint64_t> &seed = kwarg("seed", "Global program seed");
  size_t &threads =
      kwarg("threads",
            "Number of parallel eval game pairs (players swap teams) to run")
          .set_default(std::max(1u, std::thread::hardware_concurrency() - 1));
  size_t &max_games =
      kwarg("max-games", "Number of games to play before program termination")
          .set_default(1 << 14);
  bool &save = flag("save", "Sets --dir to a timestamp");
  std::string &teams_path =
      kwarg("teams", "Path to teams file").set_default("");
  std::optional<std::string> &working_dir = kwarg("dir", "Save directory");
  double &early_stop =
      kwarg("early-stop", "Forfeit when inverse sigmoid of score exceeds "
                          "this value for both players")
          .set_default(10.0);
  double &print_prob =
      kwarg("print-prob", "Probabilty to print any given battle state")
          .set_default(0);
  int &max_battle_length =
      kwarg("max-battle-length",
            "Battles exceeding this many updates are dropped")
          .set_default(-1);
  int &print_interval = kwarg("print-interval", "Seconds").set_default(15);

  size_t &buffer_size =
      kwarg("buffer-size", "Size of battle buffer (Mb) before write")
          .set_default(8);
  size_t &max_build_traj =
      kwarg("max-build-traj",
            "Size of build buffer (No. of traj's) before write")
          .set_default(1 << 10);

  std::optional<std::string> &p1_search_time_after =
      kwarg("p1-search-time-after", "");
  std::optional<std::string> &p2_search_time_after =
      kwarg("p2-search-time-after", "");
  std::optional<std::string> &p1_matrix_ucb_name_after =
      kwarg("p1-matrix-ucb-name-after", "");
  std::optional<std::string> &p2_matrix_ucb_name_after =
      kwarg("p2-matrix-ucb-name-after", "");
};

auto inverse_sigmoid(const auto x) { return std::log(x) - std::log(1 - x); }

constexpr float elo_conversion_factor = 400.0 / std::log(10);

void print(const auto &data, const bool newline = true) {
  std::cout << data;
  if (newline) {
    std::cout << '\n';
  }
}

namespace RuntimeData {
bool terminated = false;
bool suspended = false;

std::atomic<size_t> score{};
std::atomic<size_t> n{};

// filenames
std::atomic<size_t> battle_buffer_counter{};
std::atomic<size_t> build_buffer_counter{};

std::atomic<size_t> match_counter{};
std::atomic<size_t> win{};
std::atomic<size_t> loss{};
std::atomic<size_t> draw{};

std::vector<size_t> battle_lengths{};
std::vector<std::pair<MCTS::Output, MCTS::Output>> battle_outputs{};
std::atomic<size_t> thread_id{};

TeamBuilding::Provider provider;
} // namespace RuntimeData

void thread_fn(const ProgramArgs *args_ptr) {
  const auto &args = *args_ptr;
  const auto id = RuntimeData::thread_id.fetch_add(1) % args.threads;
  mt19937 device = [&args, id]() {
    mt19937 d{args.seed.value()};
    for (auto i = 0; i < id; ++i) {
      d.uniform_64();
    }
    return d.uniform_64();
  }();

  // data gen
  const size_t training_frames_target_size = args.buffer_size << 20;
  const size_t thread_frame_buffer_size = (args.buffer_size + 1) << 20;

  auto p1_battle_frame_buffer = BattleFrameBuffer{thread_frame_buffer_size};
  auto p2_battle_frame_buffer = BattleFrameBuffer{thread_frame_buffer_size};

  // These are generated slowly so a vector is fine
  std::vector<Train::Build::Trajectory> build_buffer{};

  const auto save_build_buffer_to_disk = [&build_buffer, &args]() {
    if (args.working_dir.has_value() == false) {
      return;
    }
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

  const auto play = [&](auto &p1_build_traj, auto &p2_build_traj) -> int {
    auto p1_agent =
        RuntimeSearch::Agent{.search_time = args.p1_search_time,
                             .bandit_name = args.p1_bandit_name,
                             .network_path = args.p1_network_path,
                             .discrete_network = args.p1_use_discrete,
                             .matrix_ucb_name = args.p1_matrix_ucb_name,
                             .use_table = args.p1_use_table};
    auto p1_agent_after = RuntimeSearch::Agent{
        .search_time = args.p1_search_time_after.value(),
        .bandit_name = args.p1_bandit_name,
        .network_path = args.p1_network_path,
        .discrete_network = args.p1_use_discrete,
        .matrix_ucb_name = args.p1_matrix_ucb_name_after.value(),
        .use_table = args.p1_use_table};
    auto p2_agent =
        RuntimeSearch::Agent{.search_time = args.p2_search_time,
                             .bandit_name = args.p2_bandit_name,
                             .network_path = args.p2_network_path,
                             .discrete_network = args.p2_use_discrete,
                             .matrix_ucb_name = args.p2_matrix_ucb_name,
                             .use_table = args.p2_use_table};
    auto p2_agent_after = RuntimeSearch::Agent{
        .search_time = args.p2_search_time_after.value(),
        .bandit_name = args.p2_bandit_name,
        .network_path = args.p2_network_path,
        .discrete_network = args.p2_use_discrete,
        .matrix_ucb_name = args.p2_matrix_ucb_name_after.value(),
        .use_table = args.p2_use_table};
    const auto p1_policy_options =
        RuntimePolicy::Options{.mode = args.p1_policy_mode,
                               .temp = args.p1_policy_temp,
                               .min_prob = args.p1_policy_min};
    const auto p2_policy_options =
        RuntimePolicy::Options{.mode = args.p2_policy_mode,
                               .temp = args.p2_policy_temp,
                               .min_prob = args.p2_policy_min};

    const auto &p1_team = p1_build_traj.terminal;
    const auto &p2_team = p2_build_traj.terminal;

    auto battle = PKMN::battle(p1_team, p2_team, device.uniform_64());
    auto options = PKMN::options();
    const auto result = PKMN::update(battle, 0, 0, options);
    auto input = MCTS::Input{battle, PKMN::durations(options), result};

    if (p1_agent.uses_network()) {
      p1_agent.initialize_network(battle);
      p1_agent_after.network.emplace(p1_agent.network.value());
    }
    if (p2_agent.uses_network()) {
      p2_agent.initialize_network(battle);
      p2_agent_after.network.emplace(p2_agent.network.value());
    }

    auto p1_battle_frames = Train::Battle::CompressedFrames{battle};
    auto p2_battle_frames = Train::Battle::CompressedFrames{battle};

    int p1_early_stop{}, p2_early_stop{};

    bool early_stop = false;
    size_t updates = 0;
    try {
      // playout game
      while (!pkmn_result_type(input.result)) {

        RuntimeData::battle_lengths[id] = updates;

        while (RuntimeData::suspended) {
          sleep(1);
        }

        if (RuntimeData::terminated) {
          using Ignored = int;
          return Ignored{};
        }

        const auto [p1_choices, p2_choices] =
            PKMN::choices(input.battle, input.result);

        MCTS::Output p1_output{}, p2_output{};
        int p1_index{}, p2_index{};
        p1_early_stop = 0;
        p2_early_stop = 0;
        if (p1_choices.size() > 1) {
          RuntimeSearch::Nodes nodes{};
          p1_output = RuntimeSearch::run(device, input, nodes, p1_agent);
          p1_output = RuntimeSearch::run(device, input, nodes, p1_agent_after,
                                         p1_output);
          p1_early_stop =
              inverse_sigmoid(p1_output.empirical_value) / args.early_stop;
          p1_index = process_and_sample(device, p1_output.p1_empirical,
                                        p1_output.p1_nash, p1_policy_options);
        }
        if (p2_choices.size() > 1) {
          RuntimeSearch::Nodes nodes{};
          p2_output = RuntimeSearch::run(device, input, nodes, p2_agent);
          p2_output = RuntimeSearch::run(device, input, nodes, p2_agent_after,
                                         p2_output);
          p2_early_stop =
              inverse_sigmoid(p2_output.empirical_value) / args.early_stop;
          p2_index = process_and_sample(device, p2_output.p2_empirical,
                                        p2_output.p2_nash, p2_policy_options);
        }

        RuntimeData::battle_outputs[id] = {p1_output, p2_output};

        if (updates == 0) {
          p1_build_traj.value = p1_output.empirical_value;
          p2_build_traj.value = 1 - p2_output.empirical_value;
        }

        // only if they have same sign and are both non zero
        if ((p1_early_stop * p2_early_stop) > 0) {
          // std::cout << "Early stop: " << p1_output.empirical_value << ' '
          //           << p2_output.empirical_value << std::endl;
          // std::cout << p1_early_stop << ' ' << p2_early_stop << std::endl;
          early_stop = true;
          break;
        }

        const auto p1_choice = p1_choices[p1_index];
        const auto p2_choice = p2_choices[p2_index];

        p1_battle_frames.updates.emplace_back(p1_output, p1_choice, p2_choice);
        p2_battle_frames.updates.emplace_back(p2_output, p1_choice, p2_choice);

        if (device.uniform() < args.print_prob) {
          print("GAME: " + std::to_string(RuntimeData::n.load()), false);
          print(" UPDATE: " + std::to_string(updates));
          print(PKMN::battle_data_to_string(input.battle, input.durations));
          const auto [p1_labels, p2_labels] =
              PKMN::choice_labels(input.battle, input.result);
          if (p1_choices.size() > 1) {
            print("P1:");
            MCTS::print_output(p1_output, input.battle, p1_labels, p2_labels);
          }
          if (p2_choices.size() > 1) {
            print("P2:");
            MCTS::print_output(p2_output, input.battle, p1_labels, p2_labels);
          }
        }
        input.result =
            PKMN::update(input.battle, p1_choice, p2_choice, options);
        input.durations = PKMN::durations(options);
        ++updates;
      }

      if (early_stop) {
        if (p1_early_stop > 0) {
          input.result = PKMN::result(PKMN::Result::Win);
        } else {
          input.result = PKMN::result(PKMN::Result::Lose);
        }
      }

    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
    }

    p1_battle_frames.result = input.result;
    p2_battle_frames.result = input.result;

    p1_battle_frame_buffer.write_frames(p1_battle_frames);
    p2_battle_frame_buffer.write_frames(p2_battle_frames);

    switch (pkmn_result_type(input.result)) {
    case PKMN_RESULT_WIN: {
      RuntimeData::win.fetch_add(1);
      return 2;
    }
    case PKMN_RESULT_LOSE: {
      RuntimeData::loss.fetch_add(1);
      return 0;
    }
    case PKMN_RESULT_TIE: {
      RuntimeData::draw.fetch_add(1);
      return 1;
    }
    default: {
      assert(false);
      return 0;
    }
    }
  };

  while ((args.max_games < 0) ||
         (RuntimeData::match_counter.fetch_add(1) < args.max_games)) {

    auto [p1_build_traj, p1_team_index] =
        RuntimeData::provider.get_trajectory(device);
    auto [p2_build_traj, p2_team_index] =
        RuntimeData::provider.get_trajectory(device);

    const auto s1 = play(p1_build_traj, p2_build_traj);
    const auto s2 = play(p2_build_traj, p1_build_traj);

    if (!RuntimeData::terminated) {
      RuntimeData::score.fetch_add(s1 + s2);
      RuntimeData::n.fetch_add(2);
    }
    if (args.working_dir.has_value()) {
      std::filesystem::path working_dir{args.working_dir.value()};
      if (p1_battle_frame_buffer.write_index >= training_frames_target_size) {
        p1_battle_frame_buffer.save_to_disk(working_dir / "p1",
                                            RuntimeData::battle_buffer_counter);
      }
      if (p2_battle_frame_buffer.write_index >= training_frames_target_size) {
        p2_battle_frame_buffer.save_to_disk(working_dir / "p2",
                                            RuntimeData::battle_buffer_counter);
      }
    }

    if (RuntimeData::terminated) {
      if (args.working_dir.has_value()) {
        std::filesystem::path working_dir{args.working_dir.value()};
        p1_battle_frame_buffer.save_to_disk(working_dir / "p1",
                                            RuntimeData::battle_buffer_counter);
        p2_battle_frame_buffer.save_to_disk(working_dir / "p2",
                                            RuntimeData::battle_buffer_counter);
      }
      save_build_buffer_to_disk();
      return;
    }

    if (build_buffer.size() >= args.max_build_traj) {
      save_build_buffer_to_disk();
    }
  }

  return;
}

void progress_thread_fn(const ProgramArgs *args_ptr) {
  const auto &args = *args_ptr;
  while (true) {
    for (int s = 0; s < args.print_interval; ++s) {
      if (RuntimeData::terminated) {
        return;
      }
      if ((args.max_games > 0) &&
          (RuntimeData::match_counter.load() >= args.max_games)) {
        return;
      }
      sleep(1);
    }

    const double average_score =
        RuntimeData::score.load() / 2.0 / RuntimeData::n.load();
    const double elo_difference =
        inverse_sigmoid(average_score) * elo_conversion_factor;

    std::cout << "score: " << average_score << " over " << RuntimeData::n.load()
              << " games; Elo diff: " << elo_difference << std::endl;
    std::cout << RuntimeData::win.load() << ' ' << RuntimeData::draw.load()
              << ' ' << RuntimeData::loss.load() << std::endl;

    std::cout << "info: " << std::endl;
    for (auto i = 0; i < args.threads; ++i) {
      const auto &outputs = RuntimeData::battle_outputs[i];
      std::cout << "\t" << i << ": " << RuntimeData::battle_lengths[i] << ", "
                << "(" << outputs.first.empirical_value << "/"
                << outputs.first.nash_value << "), " << "("
                << outputs.second.empirical_value << "/"
                << outputs.second.nash_value << ")";
      std::cout << std::endl;
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

void setup(auto &args) {
  if (!args.seed.has_value()) {
    args.seed.emplace(std::random_device{}());
  }
  if (!args.p1_search_time_after.has_value()) {
    args.p1_search_time_after.emplace("0ms");
  }
  if (!args.p2_search_time_after.has_value()) {
    args.p2_search_time_after.emplace("0ms");
  }
  if (!args.p1_matrix_ucb_name_after.has_value()) {
    args.p1_matrix_ucb_name_after.emplace("");
  }
  if (!args.p2_matrix_ucb_name_after.has_value()) {
    args.p2_matrix_ucb_name_after.emplace("");
  }
  // args
  if (args.save && !args.working_dir.has_value()) {
    args.working_dir.emplace("vs-" + get_current_datetime());
  }
  // create working dir
  if (args.working_dir.has_value()) {
    const std::filesystem::path working_dir = args.working_dir.value();
    std::error_code ec;
    bool created = std::filesystem::create_directory(working_dir, ec) &&
                   std::filesystem::create_directory(working_dir / "p1", ec) &&
                   std::filesystem::create_directory(working_dir / "p2", ec);
    if (ec) {
      std::cerr << "Error creating directory: " << ec.message() << '\n';
      throw std::runtime_error("Could not create working dir.");
    } else if (created) {
      std::cout << "Created directory " << working_dir.string() << std::endl;
    } else {
      throw std::runtime_error("Could not create working dir.");
    }

    // save args
    std::ofstream args_file(std::filesystem::path{working_dir} / "args");
    if (!args_file) {
      throw std::runtime_error("Failed to open args file for writing.");
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

  // stats
  RuntimeData::battle_lengths.resize(args.threads);
  RuntimeData::battle_outputs.resize(args.threads);
}

int main(int argc, char **argv) {

  std::signal(SIGINT, handle_terminate);
  std::signal(SIGTSTP, handle_suspend);

  auto args = argparse::parse<ProgramArgs>(argc, argv);

  setup(args);

  std::vector<std::thread> thread_pool{};
  for (auto t = 0; t < args.threads; ++t) {
    thread_pool.emplace_back(std::thread{&thread_fn, &args});
  }
  auto progress_thread = std::thread(&progress_thread_fn, &args);
  for (auto &thread : thread_pool) {
    thread.join();
  }
  progress_thread.join();

  std::cout << "score: "
            << (RuntimeData::score.load() / 2.0 / RuntimeData::n.load())
            << " over " << RuntimeData::n.load() << " games." << std::endl;
  std::cout << RuntimeData::win.load() << ' ' << RuntimeData::draw.load() << ' '
            << RuntimeData::loss.load() << std::endl;
}