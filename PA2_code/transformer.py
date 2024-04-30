import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(
        self, n_embd: int, head_size: int, block_size: int, dropout: float, masked: bool
    ):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.masked = masked
        self.attn_map: torch.Tensor = None

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        if self.masked:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        self.attn_map = wei
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_size: int,
        block_size: int,
        dropout: float,
        masked: bool,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(n_embd, head_size, block_size, dropout, masked)
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.attn_maps: torch.Tensor = None

    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        self.attn_maps = torch.stack([h.attn_map for h in self.heads])
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        masked: bool,
        alibi: bool,
    ):
        super().__init__()
        head_size = n_embd // n_head
        if alibi:
            self.sa = AlibiMultiHeadAttention(
                n_embd, n_head, head_size, block_size, dropout, masked
            )
        else:
            self.sa = MultiHeadAttention(
                n_embd, n_head, head_size, block_size, dropout, masked
            )
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn_maps: torch.Tensor = None

    def forward(self, x: torch.Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        self.attn_maps = self.sa.attn_maps
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float,
        n_input: int,
        n_hidden: int,
        n_output: int,
        alibi: bool = False,
    ):
        super().__init__()
        self.alibi = alibi
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) if not alibi else None
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head, block_size, dropout, masked=False, alibi=alibi)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.classifier = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )
        self.attn_maps: torch.Tensor = None

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        if self.alibi:
            x = tok_emb
        else:
            pos_emb = self.position_embedding_table(torch.arange(T))
            x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        self.attn_maps = torch.stack([block.attn_maps for block in self.blocks])
        x = x.mean(dim=1)
        logits = self.classifier(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)
        return logits, loss, self.attn_maps


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float,
        token_embedding_table: nn.Embedding = None,
        position_embedding_table: nn.Embedding = None,
        alibi: bool = False,
    ):
        super().__init__()
        self.block_size = block_size
        self.alibi = alibi
        if token_embedding_table is None:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        else:
            self.token_embedding_table = nn.Embedding.from_pretrained(
                token_embedding_table.weight, freeze=False
            )
        if position_embedding_table is None:
            self.position_embedding_table = nn.Embedding(block_size, n_embd) if not alibi else None
        else:
            self.position_embedding_table = nn.Embedding.from_pretrained(
                position_embedding_table.weight, freeze=False
            ) if not alibi else None
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head, block_size, dropout, masked=True, alibi=alibi)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.attn_maps: torch.Tensor = None

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        if self.alibi:
            x = tok_emb
        else:
            pos_emb = self.position_embedding_table(torch.arange(T))
            x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        self.attn_maps = torch.stack([block.attn_maps for block in self.blocks])

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss, self.attn_maps

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class AlibiHead(nn.Module):
    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        dropout: float,
        masked: bool,
        slope: float,
    ):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.masked = masked
        self.register_buffer("alibi_bias", self.get_alibi_biases(block_size) * slope)
        self.attn_map: torch.Tensor = None

    @torch.no_grad()
    def get_alibi_biases(self, block_size: int):
        x = torch.arange(block_size)[None, :]
        y = torch.arange(block_size)[:, None]
        return torch.tril(x - y) + torch.triu(y - x)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei += self.alibi_bias[:T, :T]
        if self.masked:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        self.attn_map = wei
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class AlibiMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_size: int,
        block_size: int,
        dropout: float,
        masked: bool,
    ):
        super().__init__()
        self.register_buffer("slopes", self.get_slopes(n_head))
        self.heads = nn.ModuleList(
            [
                AlibiHead(n_embd, head_size, block_size, dropout, masked, self.slopes[i])
                for i in range(n_head)
            ]
        )
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.attn_maps: torch.Tensor = None

    @torch.no_grad()
    def get_slopes(self, n_head: int):
        n = 2 ** math.floor(math.log2(n_head))
        m_0 = 2.0 ** (-8.0 / n)
        m = torch.pow(m_0, torch.arange(1, 1 + n))
        if n < n_head:
            m_hat_0 = 2.0 ** (-4.0 / n)
            m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_head - n), 2))
            m = torch.cat([m, m_hat])
        return m

    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        self.attn_maps = torch.stack([h.attn_map for h in self.heads])
        return out
