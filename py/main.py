import sys

import libtrain

def main ():
    if (len(sys.argv) < 2):
        print("input: buffer path")
        return
    size = 100000
    frames = libtrain.FrameInput(size)
    n = libtrain.read_buffer_to_frames(sys.argv[1], size, frames)
    print(n)
    print(libtrain.pokemon_in_dim)
    print(libtrain.active_in_dim)

    for i in range(40, 1000, 80):
        print(frames.p1_empirical[i])
        print(frames.p1_nash[i])
        print(frames.p2_empirical[i])
        print(frames.p2_nash[i])
        print()

if __name__ == "__main__":
    main()