import torch

import sys
import inspect

import libtrain
import net

class Optimizer:

    torch.nn.optimizer

def main():
    if len(sys.argv) < 3:
        print("Input: battle-file-path, buffer-size")
        return
    path = sys.argv[1]
    size = int(sys.argv[2])
    encoded_frames = libtrain.EncodedFrameInput(size)
    n_frames = libtrain.encode_buffer(
        path, size, encoded_frames, start_index=0, write_prob=0.05
    )
    print(f"read {n_frames} encoded frames")

    network = net.Network()
    output_buffer = net.OutputBuffers(size)
    net.inference(network, encoded_frames, output_buffer)


if __name__ == "__main__":
    main()
