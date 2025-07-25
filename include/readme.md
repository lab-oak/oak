## libpkmn/data/

* species.h
* moves.h
* types.h
* status.h

Clones of basic `libpkmn` types.

* strings.h

Helper functions for printing the above.

* options.h

The `LipkmnOptions` namespace defines constexpr bools for the compile-time options `-Dlog`, `-Dchance`, `-Dcalc`.


WIP
The `Options` namespace is where the user sets compile-time flags that enable various 'optimizations' throughout the project. For example, banning confusion moves means that the network inputs do not have to encode the confusion duration. All future flags are true by default and currently do nothing.

* legal-moves.h

This defines a 2d array `LEARNSETS` such that `LEARNSETS[species][move]` is true if and only if that pokemon can legally know that move at level 100. RBY does in fact have a few illegal move combinations but we simply ignore that fact; They are not competitively relevant. This file was generated by ts/legal-moves.ts and does not ever need to be re-generated.

* move-pools.h

The former array is unweildy so here we define a more natural interface e.g. `move_pool(species)`. This is also where the format options have their effect on move pools.

* layout.h

Byte offsets and sizes.

## util/

The two headers below make up the rest of the C++ `libpkmn` interface.

* data.h

Defines the `auto& View::ref(auto &)` function template that returns a reference to a wrapper for `libpkmn`'s functional types. This cheifly includes `PKMN::Battle` which provides identically named accessors for all of the zig members.

* pkmn.h

Helper fuctions to initialize Battles and Durations structs (like properly setting sleep turns and other durations.) The function aspect of `libpkmn` is wrapped here e.g. `update()`.

* debug-log.h

`libpkmn` provides a binary that can convert a buffer of battle updates into a sleak html output. This defines a struct that stores and writes the data to disk.

* sample-teams.h

Smogon sample teams for RBY OU. Two grandfathered benchmark teams are here, too.

## search/

Perfect info search essentially means open teamsheats. In fact all data is visible to the player/search besides the information described below.

Both Showdown and the cartriage games make use of hidden variables to determine the durations of sleep, confusion, disable, bide, thrashing moves, and partial-trapping moves. A counter is set when the effect is applied and it decrements from there. Only when the counter is 0 is the effect finished.
This is contrary to how most players imagine. Most players plan as though, for example, there is a chance their sleeping pokemon will wake every time it tries to use a move. While this is not correct, it is *strategically equivalent* as long as the correct probabilities are used. For any of the effects above, the 'turn-by-turn' probabilities depend on the number of turns the effect *has been active*.

```
For example, sleep is chosen uniformly from [0, 7]. This means that on the first turn a pokemon moves after sleep has been applied, the chance of wake is 1/7. The next turn it is 1/6, 1/5, 1/4, 1/3, 1/2, and finally 1/1 after 7 turns when it must wake.
```

The battle object proper does not store these duration counts, only the hidden counters. The player-visible duration info is stored in the `pkmn_gen1_chance_durations`. This is the reason for the `BattleData` struct which stores both. Only when you have both do you have a complete and natural representation of the position.

* tree.h

Search data is stored in a tree structure `Node<...>`. Each `Node` store search stats and a `std::map<Key, Node>` of its children. The keys are tuples of P1 action indices, P2 action indices, and the chance `Actions` struct. This information uniquely identifies the outcome of a turn.

This approach is probably weaker than a well-tuned transposition table but it is immune to errors like hash collision. 

* ucb.h
* exp3.h

Simultaneous bandit algorithms.

* mcts.h

The core functionality is to implement a typical MCTS algorithm where a value is estimated via the 'model' at the leaf nodes and then backpropogated through the visited nodes.

TODO lots to talk about


# train/

The structures defined here is the input and target information needed to train a value/policy model. It does not take any architecture or encoding into consideration, so a frame is the raw util/durations struct together with the value/policy targets.

Therefore the `generate` program can be used to train any network not just the one provided.

# nn/

TODO describe net architecture.

* affine.h

Basic affine layer using Eigen.

* subnet.h

Templated MLPs. Building blocks for the full net.

* embedding-cache.h

We do not need to run the full forward step every inference.

* encode.h

Write battle into inputs for Pokemon/Active nets.

* net.h

Full net with embedding-cache. Therefore each instantiation requires the 'turn 1 battle' during construction. TODO

# util/

Helpers to keep the source files tidy.