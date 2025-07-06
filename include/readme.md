## data/

A replication of `libpkmn`'s data layer for basic types. Also format specific information like legal move tables for RBY OU

## battle/

* view.h
Defines the `auto& View::ref(auto &)` function template that returns a reference to a wrapper for `libpkmn`'s functional types. This cheifly includes `View::Battle` which provides identically named accessors for all of the zig members.  

* init.h
Helper fuctions

* debug-log.h
`libpkmn` provides a binary that can convert a buffer of battle updates into a sleak html output. This defines a struct that stores and writes the data to disk.

* sample-teams.h
The Smogon sample teams for RBY OU.

## pi/

Perfect info search essentially means open teamsheats.

* tree.h

Search data is maitained with a tree of `std::unique_ptr<Node>`s that hold a `std::map< ..., std::unique_ptr<Node>>` which manages child nodes. These keys are basically `std::tuple<pkmn_choice, pkmn_choice, pkmn_chance_action>`s.

This means nodes are updated with an operator `child = node.get(i, j, obs)` that allows the tree to be traversed and grown in tandem with battle updates.

This is slower and possibly less strong than a transposition table or directed graph search but it is simpler and immune to errors. The provided bandit algorithm implementations are designed to be fast and cache-friendly.

There are two "double bandit" algorithms, UCB and Exp3 provided. It is known that stochastic algorithms like UCB 'cycle' and produce exploitable strategies in the simultaneous move setting. On the contrary, the emperical selection strategies of adversarial bandits provably converge to equilibrium. Reality often differs from theory.

* ucb.h
* exp3.h

These algorithms have an informal interface for the mcts search function.
There is an `Outcome` struct that needs to be provided with the leaf `/`.value, but is otherwise opaque.

`.update()`

`.select()`


* mcts.h

* mcts.h

