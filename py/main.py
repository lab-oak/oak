import libtrain

frames = libtrain.FrameInput(100000)

n = libtrain.read_buffer_to_frames("0.battle", 100000, frames)
# libtrain.read_battle_offsets("0.battle", 2000)
print(n)

for i in range(10):

    print(frames.p1_empirical[i])