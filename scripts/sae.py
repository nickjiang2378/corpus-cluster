"""TopK Sparse Autoencoder (OpenAI "Scaling and evaluating sparse autoencoders" style).

Features:
  - pre-bias subtraction (b_pre initialized to data mean)
  - exact TopK activation (L0 == k by construction)
  - unit-norm decoder rows, renormalized after each step
  - tied init (W_enc = W_dec.T)
  - scalar input normalization s = sqrt(d) / mean(||x||) stored in the checkpoint
  - AuxK auxiliary loss to revive dead features (model the residual with top-k_aux
    *dead* features), per OpenAI.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k, k_aux=512, aux_coef=1.0 / 32.0,
                 dead_steps_threshold=200):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.k_aux = k_aux
        self.aux_coef = aux_coef
        self.dead_steps_threshold = dead_steps_threshold

        self.b_pre = nn.Parameter(torch.zeros(d_in))
        # decoder rows: [d_sae, d_in], unit norm
        W = torch.randn(d_sae, d_in)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W.clone())
        self.W_enc = nn.Parameter(W.clone().T.contiguous())  # [d_in, d_sae] tied init
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # scalar input normalization (set via set_norm), stored as buffer
        self.register_buffer("in_scale", torch.tensor(1.0))
        # steps since each feature last fired (for dead detection)
        self.register_buffer("steps_since_fire", torch.zeros(d_sae, dtype=torch.long))

    # ---- normalization / init helpers ----
    @torch.no_grad()
    def set_norm(self, scale):
        self.in_scale.fill_(float(scale))

    @torch.no_grad()
    def set_bias_to_mean(self, mean_vec):
        self.b_pre.copy_(mean_vec * self.in_scale)

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True) + 1e-8)

    # ---- forward ----
    def encode_pre(self, x):
        return (x - self.b_pre) @ self.W_enc + self.b_enc  # [b, d_sae]

    def topk(self, pre):
        vals, idx = pre.topk(self.k, dim=-1)
        vals = F.relu(vals)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z

    def encode(self, x):
        return self.topk(self.encode_pre(x))

    def decode(self, z):
        return z @ self.W_dec + self.b_pre

    def forward(self, x, update_dead=True):
        """x already scaled by in_scale. Returns dict with recon, loss, metrics."""
        pre = self.encode_pre(x)
        z = self.topk(pre)
        recon = self.decode(z)
        recon_loss = F.mse_loss(recon, x)

        fired = (z > 0).any(dim=0)  # [d_sae]
        if update_dead:
            self.steps_since_fire += 1
            self.steps_since_fire[fired] = 0

        # AuxK: model residual with top-k_aux dead features
        aux_loss = x.new_tensor(0.0)
        dead_mask = self.steps_since_fire > self.dead_steps_threshold
        n_dead = int(dead_mask.sum())
        if self.aux_coef > 0 and n_dead > 0:
            resid = (x - recon).detach()
            pre_dead = pre.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
            kk = min(self.k_aux, n_dead)
            vals, idx = pre_dead.topk(kk, dim=-1)
            vals = F.relu(vals)
            z_aux = torch.zeros_like(pre)
            z_aux.scatter_(-1, idx, vals)
            resid_hat = z_aux @ self.W_dec  # no bias for residual
            aux_loss = F.mse_loss(resid_hat, resid)

        loss = recon_loss + self.aux_coef * aux_loss

        with torch.no_grad():
            total_var = (x - x.mean(0)).pow(2).mean()
            fvu = (recon_loss / (total_var + 1e-8)).item()
            l0 = (z > 0).float().sum(-1).mean().item()
        return {
            "recon": recon, "z": z, "loss": loss,
            "recon_loss": recon_loss.item(), "aux_loss": float(aux_loss),
            "fvu": fvu, "l0": l0, "n_dead": n_dead,
        }

    @torch.no_grad()
    def dead_fraction(self):
        return float((self.steps_since_fire > self.dead_steps_threshold).float().mean())
