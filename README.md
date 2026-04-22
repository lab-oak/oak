# About

Oak is a Python package for perfect-info search in Pokemon RBY OU with two facets:

* Program suite for data generation, network training and testing, position analysis, parameter tuning, and reinforcement learning

* Library that is powerful enough to reproduce the above as Python scripts and replace the evaluation function in existing Battle AI

# Installation

```bash
pip install oak-lab
```

Detailed instructions and caveats are first in the [tutorial](TUTORIAL.md).

# Building

**Note**: Libpkmn currently only builds with the Zig `master` release. See the [libpkmn README](https://github.com/pkmn/engine/blob/main/README.md) for more information

**Note**: Eigen has failed to build with some versions of g++-15. Try using version 14.

The project uses the GNU Multiple Precision library to solve for Nash equilibrium. It can be installed with

```bash
apt install libgmp3-dev
```

Must have Cmake and a Zig compiler installed. The script below clones the repo, builds the libpkmn libraries, and compiles the various programs in the `/src` directory

```
git clone --recurse-submodules --shallow-submodules https://github.com/lab-oak/oak
cd oak && git submodule update --recursive
chmod +x dev/libpkmn && dev/libpkmn
chmod +x dev/release && dev/release
```

If the installation fails, it is likely that the wrong version of Zig was used. The following command sets `zig` to a non-installed compiler for the duration of the terminal session. Be sure to change the path to the folder containing the zig compiler binary.

```
export PATH="/home/user/Downloads/zig-master/:$PATH"
```