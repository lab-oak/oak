#include <battle/sample-teams.h>
#include <battle/strings.h>
#include <battle/view.h>
#include <data/options.h>
#include <nn/encoding.h>
#include <nn/net.h>
#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/ucb.h>
#include <train/build-trajectory.h>
#include <train/compressed-frame.h>
#include <train/frame.h>

#include <util/random.h>

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstring>
#include <exception>
#include <format>
#include <iostream>
#include <sstream>
#include <thread>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#ifdef DEBUG
constexpr bool debug = true;
#else
constexpr bool debug = false;
#endif

void print(const auto &data, const bool newline = true) {
  if constexpr (!debug) {
    return;
  }
  std::cout << data;
  if (newline) {
    std::cout << std::endl;
  }
}

// TODO move to nn?
namespace NN {

constexpr auto build_net_hidden_dim = 512;

using BuildNet = EmbeddingNet<Encode::Team::in_dim, build_net_hidden_dim,
                              Encode::Team::out_dim, true, false>;
}; // namespace NN

// Stats for sample team matchup matrix
struct TeamPool {

  static constexpr auto n_teams = SampleTeams::teams.size();
  static constexpr auto n_entries = n_teams * (n_teams - 1) / 2;

  struct Entry {
    std::atomic<size_t> n;
    std::atomic<size_t> v;
  };

  std::array<Entry, n_entries> entries{};

  static constexpr size_t flat(auto i, auto j) {
    if (i < j) {
      return flat(j, i);
    }
    return i * (i - 1) / 2 + j;
  }

  Entry &operator()(auto i, auto j) { return entries[flat(i, j)]; }

  void update(auto i, auto j, auto score) {
    if (i < j) {
      return update(j, i, 2 - score);
    }
    auto &entry = (*this)(i, j);
    entry.n.fetch_add(1);
    entry.v.fetch_add(score);
  }
};

namespace RuntimeOptions {

size_t n_threads = 1;
size_t training_frames_target_size_mb = 8;

namespace Search {

// iterations/time per search
size_t count = 1 << 10;
char count_mode = 'i'; // (n)/(i)terations, (t)ime

// search algo, model type/path
char bandit_mode = 'e'; // (e)xp3, (u)cb
std::string battle_network_path;

// move selection
char policy_mode = 'n'; // (n)ash, (e)mpirical, (m)ix
double policy_temp = 1.0;
double policy_min_prob = 0.0; // zerod if below this threshold
double mix_nash_weight = 1.0; // (m)ix mode only

bool keep_node = true; // reuse the search tree after update
}; // namespace Search

namespace TeamGen {

// required since an untrained random network can still act as the source of
// randomness while also generating training data
std::string build_network_path;

// team gen protocal: pick from sample teams, randomly omit set data, have
// the network fill it back in and produce a trajectory using t1 search eval +
// game score as the reward
double modify_team_prob = 0;
double pokemon_delete_prob = 0;
double move_delete_prob = 0;
}; // namespace TeamGen

}; // namespace RuntimeOptions

void parse_options(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg.starts_with("--n-threads=")) {
      RuntimeOptions::n_threads = std::stoul(arg.substr(12));
    } else if (arg.starts_with("--count=")) {
      RuntimeOptions::Search::count = std::stoul(arg.substr(8));
    } else if (arg.starts_with("--count-mode=")) {
      RuntimeOptions::Search::count_mode = arg[13];
    } else if (arg.starts_with("--bandit-mode=")) {
      RuntimeOptions::Search::bandit_mode = arg[14];
    } else if (arg.starts_with("--battle-network-path=")) {
      RuntimeOptions::Search::battle_network_path = arg.substr(23);
    } else if (arg.starts_with("--policy-mode=")) {
      RuntimeOptions::Search::policy_mode = arg[14];
    } else if (arg.starts_with("--policy-temp=")) {
      RuntimeOptions::Search::policy_temp = std::stod(arg.substr(14));
    } else if (arg.starts_with("--policy-min-prob=")) {
      RuntimeOptions::Search::policy_min_prob = std::stod(arg.substr(19));
    } else if (arg.starts_with("--mix-nash-weight=")) {
      RuntimeOptions::Search::mix_nash_weight = std::stod(arg.substr(19));
    } else if (arg.starts_with("--keep-node=")) {
      RuntimeOptions::Search::keep_node =
          (arg.substr(13) == "1" || arg.substr(13) == "true");
    } else if (arg.starts_with("--build-network-path=")) {
      RuntimeOptions::TeamGen::build_network_path = arg.substr(22);
    } else if (arg.starts_with("--modify-team-prob=")) {
      RuntimeOptions::TeamGen::modify_team_prob = std::stod(arg.substr(21));
    } else if (arg.starts_with("--pokemon-delete-prob=")) {
      RuntimeOptions::TeamGen::pokemon_delete_prob = std::stod(arg.substr(24));
    } else if (arg.starts_with("--move-delete-prob=")) {
      RuntimeOptions::TeamGen::move_delete_prob = std::stod(arg.substr(21));
    } else if (arg.starts_with("--training-frames-target-size-mb=")) {
      RuntimeOptions::training_frames_target_size_mb =
          std::stoul(arg.substr(34));
    } else {
      throw std::runtime_error("Invalid arg: " + arg);
      std::cerr << "Unknown option: " << arg << "\n";
    }
  }
}

namespace RuntimeData {
bool terminated = false;
bool suspended = false;
std::string start_datetime =
    std::format("{:%F %T}\n", std::chrono::system_clock::now());
std::atomic<size_t> buffer_counter{};
std::atomic<size_t> frame_counter{};
NN::BuildNet build_network{}; // is not battle specific unlike the net
TeamPool team_pool{};
}; // namespace RuntimeData

using Obs = std::array<uint8_t, 16>;
using Exp3Node = Tree::Node<Exp3::JointBanditData<.03f, false>, Obs>;
using UCBNode = Tree::Node<UCB::JointBanditData, Obs>;

// Data that should persist only over a game - not the program or thread
// lifetime. The build network uses a game-specific cache. It is not universal
// unlike the build network.
struct SearchData {
  NN::Network battle_network;
  std::unique_ptr<Exp3Node> exp3_unique_node;
  std::unique_ptr<UCBNode> ucb_unique_node;
};

Train::BuildTrajectory
rollout_build_network(Init::Team &team, NN::BuildNet &build_net, auto &device) {
  using namespace Encode::Team;
  Train::BuildTrajectory traj{};

  auto i = 0;
  for (const auto &set : team) {
    if (set.species != Data::Species::None) {
      traj.frames[i++] = Train::ActionPolicy{species_move_table(set.species, 0), 0};
      for (const auto move : set.moves) {
        if (move != Data::Move::None) {
          traj.frames[i++] =
              Train::ActionPolicy{species_move_table(set.species, move), 0};
        }
      }
    }
  }

  std::array<float, in_dim> input{};
  write(team, input.data());
  std::array<float, out_dim> mask;
  std::array<float, out_dim> output;

  while (true) {
    mask = {};

    const bool only_requires_lead_selection =
        write_policy_mask(team, mask.data());

    build_net.propagate(input.data(), output.data());

    // softmax
    float sum = 0;
    for (auto k = 0; k < out_dim; ++k) {
      if (mask[k]) {
        output[k] = std::exp(output[k]);
        sum += output[k];
      } else {
        output[k] = 0;
      }
    }
    for (auto &x : output) {
      x /= sum;
    }

    const auto index = device.sample_pdf(output);
    traj.frames[i++] =
        Train::ActionPolicy{static_cast<uint16_t>(index), output[index]};
    input[index] = 1;
    const auto [s, m] = species_move_list(index);
    apply_index_to_team(team, s, m);

    if (only_requires_lead_selection) {
      break;
    }
  }

  return traj;
}

// produces a (modified) team from the pool and the index of unmodified team or
// -1 if modified.
std::pair<Init::Team, int>
get_team(prng &device,
         std::vector<Train::BuildTrajectory> &build_trajectories) {
  using namespace RuntimeOptions::TeamGen;

  const auto index = device.random_int(SampleTeams::teams.size());
  auto team = SampleTeams::teams[index];

  // randomly delete mons/moves for the net to fill in
  for (auto p = 0; p < 6; ++p) {
    const auto r = device.uniform();
    if (r < pokemon_delete_prob) {
      team[p] = {};
    } else {
      for (auto m = 0; m < 4; ++m) {
        const auto rm = device.uniform();
        if (rm < move_delete_prob) {
          team[p].moves[m] = Data::Move::None;
        }
      }
    }
  }

  // TODO wont produce traj if build net reconstructs a sample team

  const bool changed = (team != SampleTeams::teams[index]);
  if (changed) {
    build_trajectories.emplace_back(
        rollout_build_network(team, RuntimeData::build_network, device));
  }

  return {team, changed ? -1 : index};
}

// run search, use output to update battle data and nodes and training frame
auto search(prng &device, const BattleData &battle_data,
            SearchData &search_data, Train::CompressedFrames<> game_buffer) {
  using namespace RuntimeOptions::Search;

  auto run_search_model = [&](auto &unique_node, auto &model) {
    auto &node = *unique_node;
    MCTS search;
    if (count_mode == 'i' || count_mode == 'n') {
      return search.run(count, node, battle_data, model);
    } else if (count_mode == 't') {
      std::chrono::milliseconds ms{count};
      return search.run(ms, node, battle_data, model);
    }
    throw std::runtime_error("Invalid count mode char.");
  };

  auto run_search_node = [&](auto &unique_node) {
    if (battle_network_path.empty()) {
      MonteCarlo::Model model{device};
      return run_search_model(unique_node, model);
    } else {
      return run_search_model(unique_node, search_data.battle_network);
    }
  };

  if (bandit_mode == 'e') {
    return run_search_node(search_data.exp3_unique_node);
  } else if (bandit_mode == 'u') {
    return run_search_node(search_data.ucb_unique_node);
  }
  throw std::runtime_error("Invalid bandit mode char.");
}

std::pair<int, int> sample(prng &device, auto &output) {
  using namespace RuntimeOptions::Search;
  const double t = policy_temp;

  const auto process_and_sample = [&](const auto &policy) {
    if (t <= 0)
      throw std::runtime_error("Use positive policy power");
    std::vector<double> p(policy.begin(), policy.end());
    double sum = 0;
    for (auto &val : p) {
      val = std::pow(val, t);
      sum += val;
    }
    if (policy_min_prob > 0) {
      const double l = policy_min_prob * sum;
      sum = 0;
      for (auto &val : p) {
        if (val < l)
          val = 0;
        sum += val;
      }
    }
    for (auto &val : p)
      val /= sum;
    return device.sample_pdf(p);
  };

  if (policy_mode == 'n') {
    return {process_and_sample(output.p1_nash),
            process_and_sample(output.p2_nash)};
  } else if (policy_mode == 'e') {
    return {process_and_sample(output.p1_empirical),
            process_and_sample(output.p2_empirical)};
  } else if (policy_mode == 'm') {
    // TODO: Implement mixed mode
    throw std::runtime_error("Mix mode not implemented.");
  }
  throw std::runtime_error("Invalid policy mode.");
}

// either reset the node or swap it with its child
void update_nodes(SearchData &search_data, auto i1, auto i2, const auto &obs) {
  auto update_node = [&](auto &unique_node) {
    if (RuntimeOptions::Search::keep_node) {
      auto child = (*unique_node)[i1, i2, obs];
      if (child == unique_node->_map.end()) {
        unique_node = std::make_unique<std::decay_t<decltype(*unique_node)>>();
      } else {
        auto unique_child =
            std::make_unique<std::decay_t<decltype(*unique_node)>>(
                std::move((*child).second));
        unique_node.swap(unique_child);
      }
    } else {
      unique_node = std::make_unique<std::decay_t<decltype(*unique_node)>>();
    }
  };

  update_node(search_data.exp3_unique_node);
  update_node(search_data.ucb_unique_node);
}

// loop to generate teams, self-play battle with mcts/net, save training
// data for battle and build net
void generate(uint64_t seed) {
  prng device{seed};
  const size_t training_frames_target_size =
      RuntimeOptions::training_frames_target_size_mb << 20;
  const size_t thread_frame_buffer_size =
      (RuntimeOptions::training_frames_target_size_mb + 1) << 20;
  auto buffer = new char[thread_frame_buffer_size]{};
  size_t frame_buffer_write_index = 0;

  // These are generated slowly so a vector is fine
  std::vector<Train::BuildTrajectory> build_trajectories{};

  while (true) {
    const auto [p1_team, p1_team_index] = get_team(device, build_trajectories);
    const auto [p2_team, p2_team_index] = get_team(device, build_trajectories);

    const auto bd = Init::battle_data(p1_team, p2_team, device.uniform_64());
    BattleData battle_data{bd.first, bd.second, {}};
    pkmn_gen1_battle_options battle_options{};
    battle_data.result = Init::update(battle_data.battle, 0, 0, battle_options);

    SearchData search_data{NN::Network{}, std::make_unique<Exp3Node>(),
                           std::make_unique<UCBNode>()};
    Train::CompressedFrames<> training_frames{battle_data.battle};

    try {
      // playout game
      while (!pkmn_result_type(battle_data.result)) {

        if (RuntimeData::terminated) {
          return;
        } // TODO handle saving

        // search and sample actions
        const auto output =
            search(device, battle_data, search_data, training_frames);
        const auto [p1_index, p2_index] = sample(device, output);

        const auto [p1_choices, p2_choices] =
            Init::choices(battle_data.battle, battle_data.result);
        const auto p1_choice = p1_choices[p1_index];
        const auto p2_choice = p2_choices[p2_index];

        // update train data
        training_frames.updates.emplace_back(output, p1_choice, p2_choice);

        // update battle, durations, result (state info)
        battle_data.result = Init::update(battle_data.battle, p1_choice,
                                          p2_choice, battle_options);
        battle_data.durations =
            *pkmn_gen1_battle_options_chance_durations(&battle_options);

        // set nodes
        const auto &obs = *reinterpret_cast<const Obs *>(
            pkmn_gen1_battle_options_chance_actions(&battle_options));
        update_nodes(search_data, p1_index, p2_index, obs);
      }
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      continue;
    }

    if ((p1_team_index >= 0) && (p2_team_index >= 0)) {
      RuntimeData::team_pool.update(p1_team_index, p2_team_index,
                                    Init::score(battle_data.result));
    }

    training_frames.result = battle_data.result;
    const auto n_bytes_frames = training_frames.n_bytes();
    training_frames.write(buffer + frame_buffer_write_index);
    frame_buffer_write_index += n_bytes_frames;

    // stats
    RuntimeData::frame_counter.fetch_add(training_frames.updates.size());

    if (frame_buffer_write_index >= training_frames_target_size) {
      // write
      const auto filename =
          std::to_string(RuntimeData::buffer_counter.fetch_add(1));
      const std::string full_path =
          RuntimeData::start_datetime + "-" + filename;
      int fd = open(full_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd >= 0) {
        const auto write_result = write(fd, buffer, frame_buffer_write_index);
        close(fd);
      } else {
        std::cerr << "Failed to write buffer to " << full_path << std::endl;
      }
      // reset buffer
      std::memset(buffer, 0, thread_frame_buffer_size);
      frame_buffer_write_index = 0;
    }
  }
  delete[] buffer;
}

void print_thread_fn() {
  size_t done = 0;
  int sec = 10;
  while (true) {
    for (int i = 0; i < sec; ++i) {
      if (RuntimeData::terminated) {
        return;
      }
      sleep(1);
    }
    const auto more = RuntimeData::frame_counter.load();
    std::cout << (more - done) / (float)sec << " samples/sec." << std::endl;
    done = more;
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
  std::cout << "TERMINATED" << std::endl;
}

int main(int argc, char **argv) {
  std::signal(SIGINT, handle_terminate);
  std::signal(SIGTSTP, handle_suspend);

  parse_options(argc, argv);

  uint64_t seed = 123456789;
  int threads = RuntimeOptions::n_threads;

  prng device{seed};

  std::vector<std::thread> thread_pool;
  for (int t = 0; t < threads; ++t) {
    thread_pool.emplace_back(generate, device.uniform_64());
  }
  std::thread print_thread{print_thread_fn};

  for (auto &th : thread_pool)
    th.join();
  print_thread.join();

  return 0;
}
