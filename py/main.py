import sys
import inspect

import libtrain


def print_members_at_index(obj, n):
    for name, member in inspect.getmembers(obj):
        try:
            num = member[n].size
            if num < 300:
                print(f"{name} : {member[n]}")
            else:
                pass
        except Exception:
            pass
    for val, label in zip(obj.pokemon[n, 0, 0], libtrain.pokemon_dim_labels):
        print(val, label)


def main():
    if len(sys.argv) < 2:
        print("input: buffer path")
        return
    size = 100000
    encoded_frames = libtrain.EncodedFrameInput(size)
    n_frames = libtrain.encode_buffer(
        sys.argv[1], size, encoded_frames, start_index=0, write_prob=0.05
    )
    print(f"read {n_frames} encoded frames")

    for i in range(400, 1000, 80):
        print(f"index: {i}")
        print_members_at_index(encoded_frames, i)
        print()


if __name__ == "__main__":
    main()
