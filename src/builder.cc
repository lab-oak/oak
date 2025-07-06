#include <battle/init.h>
#include <data/legal-moves.h>
#include <data/moves.h>
#include <data/species.h>
#include <data/strings.h>
#include <nn/encoding.h>
#include <nn/subnet.h>
#include <train/build-trajectory.h>

#include <iostream>

#include <optional>

namespace NN {

constexpr auto build_net_hidden_dim = 512;

using BuildNet = EmbeddingNet<Encode::Team::in_dim, build_net_hidden_dim,
                              Encode::Team::out_dim, true, false>;
}; // namespace NN

BuildTrajectory finish_team(Init::Team &team, NN::BuildNet &build_net,
                            auto &device) {
  using namespace Encode::Team;
  BuildTrajectory traj{};

  auto i = 0;
  for (const auto &set : team) {
    if (set.species != Data::Species::None) {
      traj.frames[i++] = ActionPolicy{species_move_table(set.species, 0), 0};
      for (const auto move : set.moves) {
        if (move != Data::Move::None) {
          traj.frames[i++] =
              ActionPolicy{species_move_table(set.species, move), 0};
        }
      }
    }
  }

  std::array<float, in_dim> input{};
  write(team, input.data());
  std::array<float, out_dim> mask{};
  std::array<float, out_dim> output;

  while (true) {
    mask = {};
    bool complete = write_policy_mask(team, mask.data());

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
        ActionPolicy{static_cast<uint16_t>(index), output[index]};
    input[index] = 1;
    const auto [s, m] = species_move_list(index);
    apply_index_to_team(team, s, m);

    if (complete) {
      break;
    }
  }

  return traj;
}

int main(int argc, char **argv) {
  prng device{334234234};

  auto &build_net = *(new NN::BuildNet{});

  size_t teams = 1;

  if (argc > 1) {
    teams = std::atoll(argv[1]);
  }

  for (auto i = 0; i < teams; ++i) {

    Init::Team team{};

    const auto traj = finish_team(team, build_net, device);

    std::cout << "trajectory" << std::endl;

    for (const auto &frame : traj.frames) {
      std::cout << "action: " << frame.action
                << "; policy: " << frame.policy / 65535.0 << std::endl;
    }

    for (const auto &set : team) {
      std::cout << species_string(set.species) << ": ";
      for (auto move : set.moves) {
        std::cout << move_string(move) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  return 0;
}