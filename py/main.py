import sys
import inspect

import libtrain

def print_members_at_index(obj, n):
    for name, member in inspect.getmembers(obj):
        try:
            num = member[n].size
            if (num < 300):
                print(f"{name} : {member[n]}")
            else:
                print(f"{name} : ...")
        except Exception:
            pass

def main ():
    if (len(sys.argv) < 2):
        print("input: buffer path")
        return
    size = 100000
    encoded_frames = libtrain.EncodedFrameInput(size)
    n_frames = libtrain.encode_buffer(sys.argv[1], size, encoded_frames)
    print(f"read {n_frames} encoded frames")

    for i in range(400, 1000, 80):
        print(f"index: {i}")
        print_members_at_index(encoded_frames, i)
        break

if __name__ == "__main__":
    main()
