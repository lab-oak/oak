# libpkmn/

The `libpkmn` C API is too barebones for anything beyond the example. It is basically incumbent on the user to replicate some of the zig interface themselves. 

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