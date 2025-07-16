import ctypes
import array

import sys

lib = ctypes.CDLL('./build/libtrain.so')

pokemon_in_dim = 198
active_in_dim = 212

def initialize_encoded_buffers(size: int):
    pokemon = array.array('f',)
    pokemon.frombytes(b'\x00' * (n * pokemon_in_dim * 10 * 4))
    active = array.array('f',)
    active.frombytes(b'\x00' * (n * active_in_dim * 2 * 4))
    return pokemon, active

# Define function signature
lib.read_battle_offsets.argtypes = [
    ctypes.c_char_p,                      # path
    ctypes.POINTER(ctypes.c_uint16),     # output buffer
    ctypes.c_uint64                      # max_count
]
lib.read_battle_offsets.restype = ctypes.c_int64

# Create a buffer


# Convert array to ctypes pointer

# # Call the function
# path = b"./0.battle"
# count = 1024
# buf = array.array('H', [0] * count)  # 'H' = uint16
# n = lib.read_battle_offsets(path, buf_ptr, count)

# # Print result
# if n < 0:
#     print("Error parsing file.")
# else:
#     print(f"Got {n} entries:", buf[:n])


def main():
    if len(sys.argv) < 2:
        print("Input: buffer-path; Program will parse all battles in buffer, using the normal encoding")
        exit()
    
    buffer_path = sys.argv[1]

    count = 1024
    buf = array.array('H', [0] * count)  # 'H' = uint16
    buf_ptr = ctypes.cast(ctypes.pointer(ctypes.c_uint16.from_buffer(buf)), ctypes.POINTER(ctypes.c_uint16))
    n = lib.read_battle_offsets(bytes(buffer_path, 'utf-8'), buf_ptr, count)


if __name__ == "__main__":
    main()

