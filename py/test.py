import torch

b = 1
T = 5
N = 10
n_legal_actions = N


def get_actions():
    actions = torch.full((b, T, 1), -1, dtype=torch.int64)  # start all invalid
    policy = torch.zeros((b, T, 1))
    for i in range(b):
        start = torch.randint(0, T - 1, (1,)).item()
        done = torch.randint(start + 1, T, (1,)).item()  # done >= start
        actions[i, 0:done, 0] = torch.randint(0, N, (done,))
        policy[i, start:done, 0] = torch.rand((done - start,))
    return actions, policy


def get_state(actions):
    state = torch.zeros((b, T, N + 1), dtype=torch.float32)  # [b, T, N+1]
    safe_actions = actions + 1  # -1 -> 0, 0..N-1 -> 1..N
    state.scatter_(2, safe_actions, 1.0)
    state = torch.cumsum(state, dim=1).clamp_max(1.0)
    state = state[:, :, 1:]  # [b, T, N]
    return state

def get_mask(state):
    masks = state.clone().detach()
    masks = torch.cat([torch.zeros(b, 1, N), state[:, :-1]], dim=1)
    masks = 1 - masks
    return masks

# Random actions in [0, N), but allow -1 for invalid steps
actions, policy = get_actions()
print("actions", actions)
print("policy", policy)
state = get_state(actions)
print("state", state)
mask = get_mask(state)
print("mask", mask)

values = torch.rand([b, T, 1])  # [b, T, 1]
mask = torch.randint(0, N, (b, T, n_legal_actions), dtype=torch.int64)
logits = torch.rand([b, T, N])  # [b, T, N]
rewards = torch.rand([b, T, 1])  # [b, T, 1]

valid_actions = (actions != -1)  # [b, T, 1]
valid_choices = (policy > 0)
valid_mask = torch.logical_and(valid_actions, valid_choices).float()

# --- Log-probs over all actions ---
log_probs = torch.log_softmax(logits, dim=-1)  # [b, T, N]

# Safe gather: replace -1 with 0 (dummy index), then mask out later
safe_actions = actions.clamp(min=0)
logp_actions = torch.gather(log_probs, 2, safe_actions)  # [b, T, 1]
logp_actions = logp_actions * valid_mask  # zero out invalid

# Old log probs (pretend stored from behavior policy)
old_logp_actions = logp_actions.detach()

# --- GAE ---
gamma, lam = 0.99, 0.95
next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, -1:])], dim=1)
deltas = rewards + gamma * next_values - values

advantages = []
advantage = torch.zeros((b, 1))
for t in reversed(range(T)):
    # Only update advantage if step is valid
    advantage = deltas[:, t, :] * valid_mask[:, t, :] + gamma * lam * advantage
    advantages.insert(0, advantage)

gae = torch.stack(advantages, dim=1)  # [b, T, 1]
returns = gae + values

# --- PPO loss ---
ratios = torch.exp(logp_actions - old_logp_actions)

clip_eps = 0.2
surr1 = ratios * gae
surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * gae
policy_loss = -(
    torch.min(surr1, surr2) * valid_mask
).sum() / valid_mask.sum().clamp_min(1.0)

value_loss = (
    ((returns - values) * valid_mask) ** 2
).sum() / valid_mask.sum().clamp_min(1.0)

entropy = (
    -(log_probs * log_probs.exp()).sum(dim=-1, keepdim=True) * valid_mask
).sum() / valid_mask.sum().clamp_min(1.0)

loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

print("actions:\n", actions.squeeze(-1))
print("valid_mask:\n", valid_mask.squeeze(-1))
print("PPO loss:", loss.item())
