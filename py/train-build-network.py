import torch
import torch.nn as nn

def ppo_batch_update(
    net,           # opaque net: forward(states) -> policy_prob, value
    states,        # [batch, 31, state_dim]
    policy_probs,  # [batch, 31, 1] - old probs for taken actions
    mask,          # [batch, 31, state_dim]
    rewards,       # [batch, 31]
    gamma=0.99,
    gae_lambda=0.95,
    eps_clip=0.2,
    value_lr=1e-3,
    max_grad_norm=0.5
):
    batch, T, state_dim = states.shape
    device = states.device

    # ---------------------------------------------------
    # 1. Compute value estimates for all states
    # ---------------------------------------------------
    with torch.no_grad():
        states_flat = states.reshape(batch*T, state_dim)
        _, values_flat = net(states_flat)
        values = values_flat.view(batch, T, 1)  # [batch, 31, 1]

    # ---------------------------------------------------
    # 2. Compute GAE / returns
    # ---------------------------------------------------
    gae = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    next_value = torch.zeros(batch, 1, device=device)

    for t in reversed(range(T)):
        mask_t = (policy_probs[:, t:t+1, 0] > 0).float()  # ignore padding / uninitialized
        reward_t = rewards[:, t:t+1]
        value_t = values[:, t:t+1]
        delta = reward_t + gamma * next_value * mask_t - value_t
        gae[:, t:t+1] = delta + gamma * gae_lambda * next_value * mask_t
        returns[:, t:t+1] = gae[:, t:t+1] + value_t
        next_value = value_t

    # ---------------------------------------------------
    # 3. Flatten tensors for training
    # ---------------------------------------------------
    valid_mask = (policy_probs > 0).float()  # [batch, T, 1]
    states_train  = states.reshape(batch*T, state_dim)
    returns_train = returns.reshape(batch*T, 1)
    old_probs     = policy_probs.reshape(batch*T, 1)
    mask_train    = valid_mask.reshape(batch*T, 1)

    # ---------------------------------------------------
    # 4. Forward pass - get new probs and value
    # ---------------------------------------------------
    new_probs, values_pred = net(states_train)
    values_pred = values_pred.view(-1, 1)

    # ---------------------------------------------------
    # 5. Compute advantage
    # ---------------------------------------------------
    advantage = (returns_train - values_pred).detach()

    # ---------------------------------------------------
    # 6. Policy loss (PPO clipped)
    # ---------------------------------------------------
    # ratio = pi_new / pi_old
    ratio = (new_probs + 1e-8) / (old_probs + 1e-8)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
    policy_loss = -torch.min(surr1, surr2) * mask_train
    policy_loss = policy_loss.sum() / mask_train.sum()

    # ---------------------------------------------------
    # 7. Value loss
    # ---------------------------------------------------
    value_loss = ((values_pred - returns_train)**2 * mask_train).sum() / mask_train.sum()

    # ---------------------------------------------------
    # 8. Backward / optimizer step
    # ---------------------------------------------------
    net.optimizer.zero_grad()
    (policy_loss + 0.5 * value_loss).backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
    net.optimizer.step()

    return policy_loss.item(), value_loss.item(), returns

