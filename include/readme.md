# libpkmn/

The `libpkmn` C API is too minimal to use for a project of this scale. Only the outmost types like `Battle` and `Options` are exponsed, and only as raw bytes.  
We are forced to reimplement much of the libraries zig code in C++.

## libpkmn/data/

Basic types

# data/

Format specific information, teams

#  search/

Perfect info search code

#  train/

Value and policy targets, not specific to any network

# encode/

Oak specific battle and team-gen network encodings

# nn/

NN implementation. Full precision and quantized