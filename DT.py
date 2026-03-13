import torch
import torch.nn as nn

# Default model hyperparameters
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1


class DecisionTransformer(nn.Module):
    """Decision Transformer with token-type embedding and auxiliary reward head.

    Forward returns action predictions by default.
    Set return_aux=True to also get reward predictions.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=n_embd,
        max_length=200,
        num_layers=n_layer,
        num_heads=n_head,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Input embeddings
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(action_dim, hidden_dim)
        self.embed_rtg = nn.Linear(1, hidden_dim)
        self.embed_timestep = nn.Embedding(max_length, hidden_dim)
        self.embed_token_type = nn.Embedding(3, hidden_dim)  # 0: RTG, 1: STATE, 2: ACTION

        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
        )
        try:
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.predict_reward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Zero-init keeps backward behavior stable when loading old checkpoints with strict=False.
        with torch.no_grad():
            self.embed_token_type.weight.zero_()

    def forward(self, states, actions, rtgs, timesteps, attention_mask=None, return_aux=False):
        """
        states: [B, K, state_dim]
        actions: [B, K, action_dim]
        rtgs: [B, K, 1]
        timesteps: [B, K]
        attention_mask: [B, K] (1 for valid, 0 for padded)
        """
        bsz, k, _ = states.shape
        timesteps = timesteps.clamp(min=0, max=self.max_length - 1)

        time_emb = self.embed_timestep(timesteps)
        type_emb = self.embed_token_type(torch.arange(3, device=states.device))

        state_emb = self.embed_state(states) + time_emb + type_emb[1]
        action_emb = self.embed_action(actions) + time_emb + type_emb[2]
        rtg_emb = self.embed_rtg(rtgs) + time_emb + type_emb[0]

        stacked_valid_mask = None
        if attention_mask is not None:
            valid_mask = attention_mask.bool()
            token_mask = valid_mask.unsqueeze(-1).float()
            state_emb = state_emb * token_mask
            action_emb = action_emb * token_mask
            rtg_emb = rtg_emb * token_mask
            stacked_valid_mask = torch.stack((valid_mask, valid_mask, valid_mask), dim=2).view(bsz, 3 * k)

        # [R1,s1,a1, R2,s2,a2, ...]
        x = torch.stack((rtg_emb, state_emb, action_emb), dim=2).view(bsz, 3 * k, self.hidden_dim)
        x = self.embed_dropout(self.embed_ln(x))

        # Causal attention over token sequence
        causal_mask = torch.triu(
            torch.ones(3 * k, 3 * k, device=states.device, dtype=torch.bool),
            diagonal=1,
        )
        # NOTE:
        # In eval/no_grad with heavy left-padding, passing src_key_padding_mask can trigger
        # NaN outputs in some optimized attention kernels when padded queries become all-masked.
        # We keep padding tokens zeroed above and only use causal mask here for stability.
        h = self.transformer(x, mask=causal_mask)
        h = self.final_ln(h)

        state_hidden = h[:, 1::3, :]
        action_preds = self.predict_action(state_hidden)

        if return_aux:
            reward_preds = self.predict_reward(state_hidden)
            return action_preds, reward_preds
        return action_preds
