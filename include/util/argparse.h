#pragma once

#include "argparse/argparse.hpp"

struct TeamBuildingArgs : public argparse::Args {
  double &team_modify_prob =
      kwarg("team-modify-prob", "Probability the base team (from sample teams "
                                "or teams file) is modified")
          .set_default(0);
  double &pokemon_delete_prob =
      kwarg("pokemon-delete-prob",
            "Probability a set (species/moves) are omitted")
          .set_default(0);
  double &move_delete_prob =
      kwarg("move-delete-prob", "Probability a move is deleted").set_default(0);
  std::string &build_network_path =
      kwarg("build-network-path", "").set_default("");
  int &max_pokemon = kwarg("max-pokemon", "Max team size").set_default(6);
};

#define MAKE_AGENT_ARGS(NAME, BASE, A, B)                                      \
  struct NAME : public BASE {                                                  \
    std::string &A##search_time =                                              \
        kwarg(B "search-time", "Default search time").set_default("4096");     \
                                                                               \
    std::optional<std::string> &A##fast_search_time =                          \
        kwarg(B "fast-search-time", "Search time when using a quick search");  \
                                                                               \
    std::optional<std::string> &A##t1_search_time =                            \
        kwarg(B "t1-search-time", "Search time on turn 0");                    \
                                                                               \
    std::string &A##bandit_name =                                              \
        kwarg(B "bandit-name", "Bandit algorithm").set_default("ucb-0.173");   \
                                                                               \
    std::string &A##network_path =                                             \
        kwarg(B "network-path", "Network path").set_default("mc");             \
                                                                               \
    bool &A##use_discrete =                                                    \
        flag(B "use-discrete", "Enable Stockfish discrete main subnet");       \
                                                                               \
    char &A##policy_mode =                                                     \
        kwarg(B "policy-mode", "Policy mode").set_default('e');                \
                                                                               \
    double &A##policy_temp =                                                   \
        kwarg(B "policy-temp", "P-norm when using (e)mpirical mode")           \
            .set_default(2.5);                                                 \
                                                                               \
    double &A##policy_min =                                                    \
        kwarg(B "policy-min", "Probs below this will be zerod")                \
            .set_default(.001);                                                \
  };

MAKE_AGENT_ARGS(AgentArgs, argparse::Args, , "")
MAKE_AGENT_ARGS(SelfPlayAgentArgs, TeamBuildingArgs, , "")
MAKE_AGENT_ARGS(P1AgentArgs, TeamBuildingArgs, p1_, "p1-")
MAKE_AGENT_ARGS(VsAgentArgs, P1AgentArgs, p2_, "p2-")