import os

import read


def find_battle_files(root_dir):
    battle_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".battle"):
                full_path = os.path.join(dirpath, filename)
                battle_files.append(full_path)
    return battle_files

def main():
    total = 0   
    files = find_battle_files("/home/user/train/2025-07-14-18:30:58/")
    for file in files:
        battle_data = read.read_battle_data(file)
        # battle_frames = [libread.get_frames(*pee) for pee in battle_data]        # battle_frames = [libread.get_frames(*pee) for pee in battle_data]
        battle_frame_sizes = sum([n for _, n in battle_data])
        total += battle_frame_sizes
        # for frame in battle_frames:
        #     total += frame.size
    print(total)


if __name__ == "__main__":
    main()  
