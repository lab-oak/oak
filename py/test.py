import torch

b = 1
T = 5
N = 10
n_legal_actions = 3

# Random actions in [0, N)
actions = torch.randint(0, N, (b, T, 1), dtype=torch.int64)
policy = torch.rand([b, T, 1])

# Mock NN outputs
values = torch.rand([b, T, 1])                  # [b, T, 1]
mask = torch.randint(0, N, (b, T, n_legal_actions), dtype=torch.int64)  # [b, T, n_legal_actions]
logits = torch.rand([b, T, N])                  # [b, T, N]
rewards = torch.rand([b, T, 1])                 # [b, T, 1]


# One-hot for each step
one_hot = torch.zeros([b, T, N], dtype=torch.float32)
one_hot.scatter_(2, actions, 1.0)

# Cumulative rollout: keep past actions "on"
state = torch.cumsum(one_hot, dim=1).clamp_max(1.0)

# --- Masked logits (for legal actions only) ---
masked_logits = torch.gather(logits, 2, mask)   # [b, T, n_legal_actions]
print(actions)
print(logits)
print(masked_logits)

# --- Log-probs over ALL actions ---
log_probs = torch.log_softmax(logits, dim=-1)    # [b, T, N]

print("log probs", log_probs)

# Chosen action log-probs (direct gather)
logp_actions = torch.gather(log_probs, 2, actions)  # [b, T, 1]
print("log p actions", logp_actions)

# Old log probs (pretend stored from behavior policy)
old_logp_actions = torch.log(policy)

# --- GAE ---
gamma, lam = 0.99, 0.95
next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, -1:])], dim=1)
deltas = rewards + gamma * next_values - values  # [b, T, 1]

advantages = []
advantage = torch.zeros((b, 1))  # [b, 1]
for t in reversed(range(T)):
    advantage = deltas[:, t, :] + gamma * lam * advantage  # [b, 1]
    advantages.insert(0, advantage)
    print("advantage", t)
    print(advantage)

gae = torch.stack(advantages, dim=1)  # [b, T, 1]
returns = gae + values

# --- PPO loss ---
ratios = torch.exp(logp_actions - old_logp_actions)

clip_eps = 0.2
surr1 = ratios * gae
surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * gae
policy_loss = -torch.min(surr1, surr2).mean()

value_loss = (returns - values).pow(2).mean()

entropy = -(log_probs * log_probs.exp()).sum(dim=-1, keepdim=True).mean()

loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

print("log_probs shape:", log_probs.shape)        # [b, T, N]
print("logp_actions shape:", logp_actions.shape)  # [b, T, 1]
print("gae shape:", gae.shape)                    # [b, T, 1]
print("PPO loss:", loss.item())
