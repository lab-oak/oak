# About

This project has two layers. The first is a header-only library that defines

* a C++ interface for `libpkmn` that closely replicates the feel of using the library in native Zig

* *sound*, high performance MCTS for perfect information battles

* both full-precision and fast, quantized (TODO) neural network implementations for CPU

* a lossless battle encoding for neural network inputs

* highly compressed disk formats for battle and build network training data

The second layer is a collection of programs that form a serious platorm for training and evaluating neural networks for battling and team building. These include:

* a script for generating self-play training data

* a static library to quicly convert compressed training data into Pytorch tensors

* a tool to compare battle networks with each other and pure MCTS

* a tool for engine analysis of user defined positions

More information about the header-library is in [here](include/readme.md)

Information about the programs can be found by running them with `--help`.

# Building

Must have cmake and zig installed. The bash below clones the repo, builds the lipkmn libraries, and compiles the various programs in the `/src` directory

```
git clone --recurse-submodules https://github.com/lab-oak/oak
cd oak && git submodule update --recursive
chmod +x dev/libpkmn && ./dev/libpkmn
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make && cd ..
```

The `pkmn-debug` utility is built via
```
cd extern/engine
npm install && npm run compile
```
and run with 
```
./extern/engine/src/bin/pkmn-debug
```
