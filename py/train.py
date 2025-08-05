import sys
import os

if len(sys.argv) < 4:
    print("Usage: net-dir-name battle-file-path buffer-size")
    exit(1)

import torch

import libtrain
import net


def find_battle_files(root_dir):
    battle_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".battle"):
                full_path = os.path.join(dirpath, filename)
                battle_files.append(full_path)
    return battle_files


class Optimizer:
    def __init__(self, network: torch.nn.Module, lr):
        self.opt = torch.optim.Adam(network.parameters(), lr=lr)

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()


def kl_div(logit, target):
    log_probs = torch.log_softmax(logit, dim=-1)
    kl = target * (torch.log(target) - log_probs)
    kl = kl.masked_fill(torch.isneginf(logit), 0)
    return kl.sum(dim=1).mean(dim=0)


def loss(input: libtrain.EncodedFrame, output: net.OutputBuffers, print_flag=False):
    size = min(input.size, output.size)

    w_nash = 0.0
    w_empirical = 1.0
    w_score = 0.0

    value_target = (
        w_nash * input.nash_value[:size]
        + w_empirical * input.empirical_value[:size]
        + w_score * input.score[:size]
    )

    w_nash_p = 1
    w_empirical_p = 1 - w_nash_p

    p1_policy_target = (
        w_empirical_p * input.p1_empirical[:size] + w_nash_p * input.p1_nash[:size]
    )
    p2_policy_target = (
        w_empirical_p * input.p2_empirical[:size] + w_nash_p * input.p2_nash[:size]
    )

    p1 = output.p1_policy[:size]
    p2 = output.p2_policy[:size]

    p1_loss = kl_div(p1, p1_policy_target)
    p2_loss = kl_div(p2, p2_policy_target)
    value_loss = torch.nn.functional.mse_loss(output.value[:size], value_target)

    loss = value_loss + p1_loss + p2_loss

    if print_flag:
        window = 5
        x = torch.nn.functional.softmax(p1[:window], 1).view(window, 1, 9)
        y = p1_policy_target[:window].view(window, 1, 9)
        print(
            torch.cat(
                [
                    output.value[:window],
                    value_target[:window],
                    input.empirical_value[:window],
                    input.nash_value[:window],
                    input.score[:window],
                ],
                dim=1,
            )
        )
        print(torch.cat([x, y], dim=1))
        print(f"loss: v:{value_loss.mean()}, p1:{p1_loss}, p2:{p2_loss}")

    return loss


def main():
    threads = 4
    steps = 2**16
    checkpoint = 500

    net_dir = sys.argv[1]
    parent = sys.argv[2]
    size = int(sys.argv[3])

    if os.path.exists(net_dir):
        print(f"Folder {net_dir} already exists. Please specify a new dir for work.")
        return
    os.makedirs(net_dir, exist_ok=False)

    paths = find_battle_files(parent)
    print(f"{len(paths)} paths found")

    encoded_frames = libtrain.EncodedFrame(size)
    output_buffer = net.OutputBuffers(size)
    network = net.Network()
    optimizer = Optimizer(network, 0.001)

    for step in range(steps):
        encoded_frames.clear()
        output_buffer.clear()

        n_frames = libtrain.encode_buffers(
            paths, threads, size, encoded_frames, start_index=0, write_prob=1 / 100
        )

        network.inference(encoded_frames, output_buffer)

        optimizer.zero_grad()
        loss_value = loss(encoded_frames, output_buffer, (step % checkpoint) == 0)
        loss_value.backward()
        optimizer.step()

        network.clamp_parameters()

        if step % checkpoint == 0:
            ckpt_path = os.path.join(net_dir, f"{step}.net")
            with open(ckpt_path, "wb") as f:
                network.write_parameters(f)
            print(f"Checkpoint saved at step {step}: {ckpt_path}")

    final_path = os.path.join(net_dir, f"{steps}.net")
    with open(final_path, "wb") as f:
        network.write_parameters(f)
    print(f"Final network saved: {final_path}")


if __name__ == "__main__":
    main()
