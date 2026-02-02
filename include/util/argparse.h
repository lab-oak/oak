#pragma once

#include "argparse/argparse.hpp"

namespace Argparse {
template <typename T> using Identity = T;
}

struct TeamBuildingArgs : public argparse::Args {
  double &team_modify_prob =
      kwarg("team-modify-prob", "Probability the base team (from sample teams "
                                "or teams file) is modified")
          .set_default(0);
  double &pokemon_delete_prob =
      kwarg("pokemon-delete-prob",
            "Probability a set (species + moveset) is omitted")
          .set_default(0);
  double &move_delete_prob =
      kwarg("move-delete-prob", "Probability a move is omitted").set_default(0);
  std::string &build_network_path =
      kwarg("build-network-path", "").set_default("");
  int &max_pokemon = kwarg("max-pokemon", "Max team size").set_default(6);
};

#define MAKE_AGENT_ARGS(NAME, BASE, WRAPPER, A, B)                             \
  struct NAME : public BASE {                                                  \
    WRAPPER<std::string> &A##search_time =                                     \
        kwarg(B "search-time", "Default search time");                         \
                                                                               \
    WRAPPER<std::string> &A##bandit_name =                                     \
        kwarg(B "bandit-name", "Bandit algorithm");                            \
                                                                               \
    WRAPPER<std::string> &A##matrix_ucb_name =                                 \
        kwarg(B "matrix-ucb-name", "MatrixUCB start/interval/minimum/c")       \
            .set_default("");                                                  \
                                                                               \
    WRAPPER<std::string> &A##network_path =                                    \
        kwarg(B "network-path", "Network path").set_default("mc");             \
                                                                               \
    bool &A##use_discrete =                                                    \
        flag(B "use-discrete", "Enable Stockfish discrete main subnet");       \
                                                                               \
    bool &A##use_table =                                                       \
        flag(B "use-table", "Use a transposition table instead of a tree");    \
  };

#define MAKE_AGENT_POLICY_ARGS(NAME, BASE, WRAPPER, A, B)                      \
  struct NAME : public BASE {                                                  \
    WRAPPER<char> &A##policy_mode = kwarg(B "policy-mode", "Policy mode");     \
    double &A##policy_temp =                                                   \
        kwarg(B "policy-temp", "P-norm just before clipping/sampling")         \
            .set_default(1);                                                   \
    double &A##policy_nash_weight =                                            \
        kwarg(B "policy-nash-weight",                                          \
              "Weight of nash policy when using (m)ixed mode")                 \
            .set_default(.5);                                                  \
    double &A##policy_min =                                                    \
        kwarg(B "policy-min", "Probs below this will be zerod")                \
            .set_default(0);                                                   \
  };

MAKE_AGENT_POLICY_ARGS(AgentPolicyArgs, TeamBuildingArgs, Argparse::Identity, ,
                       "")
MAKE_AGENT_POLICY_ARGS(FastAgentPolicyArgs, AgentPolicyArgs, std::optional,
                       fast_, "fast-")
MAKE_AGENT_ARGS(AgentArgs, FastAgentPolicyArgs, Argparse::Identity, , "")
MAKE_AGENT_ARGS(T1AgentArgs, AgentArgs, std::optional, t1_, "t1-")
MAKE_AGENT_ARGS(FastAgentArgs, T1AgentArgs, std::optional, fast_, "fast-")
MAKE_AGENT_ARGS(AfterAgentArgs, FastAgentArgs, std::optional, after_, "after-")
using GenerateAgentArgs = AfterAgentArgs;

MAKE_AGENT_POLICY_ARGS(P1PolicyArgs, TeamBuildingArgs, Argparse::Identity, p1_,
                       "p1-")
MAKE_AGENT_POLICY_ARGS(P2PolicyArgs, P1PolicyArgs, Argparse::Identity, p2_,
                       "p2-")
MAKE_AGENT_ARGS(P1AgentArgs, P2PolicyArgs, Argparse::Identity, p1_, "p1-")
MAKE_AGENT_ARGS(P2AgentArgs, P1AgentArgs, Argparse::Identity, p2_, "p2-")
using VsAgentArgs = P2AgentArgs;