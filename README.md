# About

Oak is a software toolkit for perfect info search and neural network training for the first generation of Pokemon (RBY). It includes:

* A program for generating neural network training data for both battling and team building

* An analysis engine that accepts any position in a a simple text format

* A tool for comparing the strength of different neural network and search parameters

* A python library for quickly reading training data into numpy arrays

This is all is intended to work out of the box to allow people with no programming knowledge to compile training data and train neural networks in Python

These programs were all built from a shared header library that is easy to extend and modify. This library includes:

* a complete C++ interface for `libpkmn` that mirrors the original Zig code

* correct, high-performance mcts implementations

* simple neural networks for value/policy inference and team generation

* compressed training data formats for battling and team generation.

# Building

**Note**: Libpkmn currently fails to build with the latest Zig. Use the official v0.11 release

**Note**: Eigen has failed to build with some versions of g++-15.

Must have Cmake and a Zig compiler installed. The script below clones the repo, builds the lipkmn libraries, and compiles the various programs in the `/src` directory

```
git clone --recurse-submodules --shallow-submodules https://github.com/lab-oak/oak
cd oak && git submodule update --recursive
chmod +x dev/libpkmn && ./dev/libpkmn
mkdir release && cd release && cmake .. -DCMAKE_BUILD_TYPE=Release && make && cd ..
```

If the installation fails, it is likely that the wrong version of Zig was used. The following command sets `zig` to the a non-installed compiler for the duration of the terminal session. Be sure to change the path to the folder containing the zig compiler binary.

```
export PATH="/home/user/Downloads/zig11/:$PATH"
```