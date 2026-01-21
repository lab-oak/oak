# About

Oak is a software toolkit for perfect info search and neural network training for the first generation of Pokemon (RBY). It includes:

* Program for generating neural network training data for both battling and team building

* Python scripts for training battling and team-building networks using the data from above 

* Tool for comparing the strength of the trained networks and comparing them to Monte-Carlo and Poke-Engine baselines

* Analysis engine that accepts arbitrary game states in a simple text format

* Reinforcement Learning master script that configures and runs the data generation and learning scripts concurrently

This is all is intended to work out of the box to allow people with no programming knowledge to compile training data and train neural networks in Python

These programs were all built from a shared header library that is easy to extend and modify. This library includes:

* Complete C++ interface for `libpkmn` that mirrors the original Zig code

* High-performance MCTS implementation with a litany of variations

* Eigen backed neural networks for value/policy inference and team generation

* Compressed training data formats for battling and team generation.

# Building

**Note**: Libpkmn currently only builds with the Zig `master` release. See the Libpkmn README for more information

**Note**: Eigen has failed to build with some versions of g++-15. Try using version 14.

The project uses the GNU Multiple Precision library to solve for Nash equilibrium. It can be installed with

```
sudo apt install libgmp3-dev
```

Must have Cmake and a Zig compiler installed. The script below clones the repo, builds the lipkmn libraries, and compiles the various programs in the `/src` directory

```
git clone --recurse-submodules --shallow-submodules https://github.com/lab-oak/oak
cd oak && git submodule update --recursive
chmod +x dev/libpkmn && ./dev/libpkmn
mkdir release && cd release && cmake .. -DCMAKE_BUILD_TYPE=Release && make && cd ..
```

If the installation fails, it is likely that the wrong version of Zig was used. The following command sets `zig` to the a non-installed compiler for the duration of the terminal session. Be sure to change the path to the folder containing the zig compiler binary.

```
export PATH="/home/user/Downloads/zig-master/:$PATH"
```