#pragma once

#include <encode/team.h>
#include <format/move-pools.h>
#include <nn/build-network.h>
#include <train/build-trajectory>

// Rollout battle network

auto rollout(auto &network, const auto &team) {

  Train::BuildTrajectory trajectory;

  trajectory.initial = {team.begin(), team.end()};

  const auto n = team.size();

  struct Helper {
    bool needs_species{};
    uint move_index{};
    uint max_moves{};
    Format::MovePool move_pool;

    Helper(const auto &set)
        : needs_species{!static_cast<bool>(set.species)},
          max_moves{Format::move_pool_size(set.species)},
          move_index{
              std::count_if(set.moves.begin(), set.moves.end(),
                            [](auto m) { return static_cast<bool>(m.id); })},
          move_pool{Format::move_pool(set.species)} {}

    bool complete() const { return !needs_species && move_index >= max_moves; }
  };

  std::vector<Helper> helpers{};
  helpers.resize(n);
  std::transform(team.begin(), team.end(), helpers.begin(),
                 [](const auto &set) { return set; });

  while (std::any_of(helpers.begin(), helpers.end(),
                     [](const auto &s) { return !s.complete(); })) {


    
    

  }
}