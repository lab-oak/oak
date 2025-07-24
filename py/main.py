import sys

import libtrain

def main ():
    if (len(sys.argv) < 2):
        print("input: buffer path")
        return
    size = 100000
    encoded_frames = libtrain.EncodedFrameInput(size)
    n_frames = libtrain.encode_buffer(sys.argv[1], size, encoded_frames)
    print(f"{n_frames} encoded frames")

    for i in range(40, 1000, 80):
        print(encoded_frames.p1_empirical[i])
        print(encoded_frames.p1_nash[i])
        print(encoded_frames.p2_empirical[i])
        print(encoded_frames.p2_nash[i])
        print()

if __name__ == "__main__":
    main()