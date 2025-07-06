# Building

Must have cmake and zig installed. The bash below clones the repo, builds the lipkmn libraries, and compiles the various programs in the `/src` directory

```
git clone --recurse-submodules https://github.com/lab-oak/oak
cd oak && git submodule update --recursive
chmod +x dev/libpkmn && ./dev/libpkmn
mkdir build && cd build && cmake .. && make && cd ..
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
