import torch
import torch.nn as nn

# Keep defaults aligned with DT.py for fair comparison.
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1


class ConstrainedDecisionTransformer(nn.Module):
    """
    CDT variant with explicit constraint token:
    token order per step: [CTG_t, RTG_t, S_t, A_t]

    - CTG: constraint-to-go (surrogate waiting cost budget)
    - RTG: return-to-go
    - S/A: state/action
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        constraint_dim=1,
        hidden_dim=n_embd,
        max_length=200,
        num_layers=n_layer,
        num_heads=n_head,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.constraint_dim = int(constraint_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_length = int(max_length)

        self.embed_state = nn.Linear(self.state_dim, self.hidden_dim)
        self.embed_action = nn.Linear(self.action_dim, self.hidden_dim)
        self.embed_rtg = nn.Linear(1, self.hidden_dim)
        self.embed_ctg = nn.Linear(self.constraint_dim, self.hidden_dim)
        self.embed_timestep = nn.Embedding(self.max_length, self.hidden_dim)
        # 0: CTG, 1: RTG, 2: STATE, 3: ACTION
        self.embed_token_type = nn.Embedding(4, self.hidden_dim)

        self.embed_ln = nn.LayerNorm(self.hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.hidden_dim,
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
        self.final_ln = nn.LayerNorm(self.hidden_dim)

        self.predict_action = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.predict_reward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.predict_cost = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.constraint_dim),
        )

        with torch.no_grad():
            self.embed_token_type.weight.zero_()

    def forward(
        self,
        states,
        actions,
        rtgs,
        ctgs,
        timesteps,
        attention_mask=None,
        return_aux=False,
    ):
        """
        states:   [B, K, state_dim]
        actions:  [B, K, action_dim]
        rtgs:     [B, K, 1]
        ctgs:     [B, K, constraint_dim]
        timesteps:[B, K]
        """
        bsz, k, _ = states.shape
        timesteps = timesteps.clamp(min=0, max=self.max_length - 1)

        time_emb = self.embed_timestep(timesteps)
        type_emb = self.embed_token_type(torch.arange(4, device=states.device))

        ctg_emb = self.embed_ctg(ctgs) + time_emb + type_emb[0]
        rtg_emb = self.embed_rtg(rtgs) + time_emb + type_emb[1]
        state_emb = self.embed_state(states) + time_emb + type_emb[2]
        action_emb = self.embed_action(actions) + time_emb + type_emb[3]

        if attention_mask is not None:
            valid_mask = attention_mask.bool().unsqueeze(-1).float()
            ctg_emb = ctg_emb * valid_mask
            rtg_emb = rtg_emb * valid_mask
            state_emb = state_emb * valid_mask
            action_emb = action_emb * valid_mask

        # [CTG1, RTG1, S1, A1, CTG2, RTG2, S2, A2, ...]
        x = torch.stack((ctg_emb, rtg_emb, state_emb, action_emb), dim=2).view(
            bsz, 4 * k, self.hidden_dim
        )
        x = self.embed_dropout(self.embed_ln(x))

        causal_mask = torch.triu(
            torch.ones(4 * k, 4 * k, device=states.device, dtype=torch.bool),
            diagonal=1,
        )
        h = self.transformer(x, mask=causal_mask)
        h = self.final_ln(h)

        state_hidden = h[:, 2::4, :]
        action_preds = self.predict_action(state_hidden)

        if not return_aux:
            return action_preds

        reward_preds = self.predict_reward(state_hidden)
        cost_preds = self.predict_cost(state_hidden)
        return action_preds, reward_preds, cost_preds
