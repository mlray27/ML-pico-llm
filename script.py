# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gc
import os

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.END_TOKEN_ID = 50256

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        ##################### add special_token ###################
        
        seq = seq + [self.END_TOKEN_ID]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, embed_size)


        layers = []
        input_dim = k * embed_size
        hidden_dim = embed_size
        output_dim = vocab_size
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(num_inner_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        seq_len, batch_size = tokens_seq.shape
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_tensor = torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device)
                    context_embed = self.embedding(context_tensor)  # (k, embed_size)
                    context_flat = context_embed.flatten().unsqueeze(0)  # (1, k * embed_size)

                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs

################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################
def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).reshape_as(x)

    
def apply_rotary_pos_emb(q, k, sin, cos):


    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    q_rotated = torch.cat([
        q1 * cos - q2 * sin,
        q1 * sin + q2 * cos
    ], dim=-1)

    k_rotated = torch.cat([
        k1 * cos - k2 * sin,
        k1 * sin + k2 * cos
    ], dim=-1)

    return q_rotated, k_rotated

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, dim/2)

        self.register_buffer("sin", torch.sin(freqs).unsqueeze(0), persistent=False)  # (1, seq_len, dim/2)
        self.register_buffer("cos", torch.cos(freqs).unsqueeze(0), persistent=False)

    def forward(self, seq_len, device):
        return (
            self.sin[:, :seq_len, :].to(device),
            self.cos[:, :seq_len, :].to(device)
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads ==0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model//n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024)
        )

        self.k_cache = None
        self.v_cache = None

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=2048)

    def forward(self, x, use_cache=False):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        
        sin, cos = self.rope(T, x.device)

        sin = sin.unsqueeze(1)  # (1, 1, T, hd/2)
        cos = cos.unsqueeze(1)  # (1, 1, T, hd/2)
        
        q, k = apply_rotary_pos_emb(q, k, sin, cos)
        
        if use_cache and self.k_cache is not None and self.v_cache is not None:
            if self.k_cache.device != k.device:
                self.k_cache = self.k_cache.to(k.device)
                self.v_cache = self.v_cache.to(v.device)
                
            k_all = torch.cat([self.k_cache, k], dim=2)
            v_all = torch.cat([self.v_cache, v], dim=2)
            
            self.k_cache = k_all
            self.v_cache = v_all
        else:
            k_all = k
            v_all = v
            
            if use_cache:
                self.k_cache = k
                self.v_cache = v

        S = k_all.size(2)
        att = (q @ k_all.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, nh, T, S)
        
        if use_cache and S > T:
            cache_len = S-T
            att_mask = torch.ones(T, S, device=x.device)
            for i in range(T):
                att_mask[i, cache_len+i+1:] = 0 
            att_mask = att_mask.view(1, 1, T, S)
            att = att.masked_fill(att_mask == 0, float('-inf'))
        else:
            att = att.masked_fill(self.mask[:, :, :T, :S] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        
        y = att @ v_all  # (B, nh, T, hd)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.out_proj(y)
        
        return y
    
    # def forward(self, x, use_cache=False):
    #     B, T, C = x.shape

    #     q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)# (B, nh, T, hd)
    #     k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)# (B, nh, T, hd)
    #     v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)# (B, nh, T, hd)
        
    #     if use_cache and self.k_cache is not None and self.v_cache is not None:
    #         if self.k_cache.device != k.device:
    #             self.k_cache = self.k_cache.to(k.device)
    #             self.v_cache = self.v_cache.to(v.device)
                
    #         k_all = torch.cat([self.k_cache, k], dim=2)
    #         v_all = torch.cat([self.v_cache, v], dim=2)
            
    #         self.k_cache = k_all
    #         self.v_cache = v_all
    #     else:
    #         k_all = k
    #         v_all = v
            
    #         if use_cache:
    #             self.k_cache = k
    #             self.v_cache = v


    #     S = k_all.size(2)
    #     att = (q @ k_all.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))# (B, nh, T, S)
        
    #     if use_cache and S>T:
    #         cache_len = S-T
    #         att_mask = torch.ones(T, S, device=x.device)
    #         for i in range(T):
    #             att_mask[i, cache_len+i+1:] = 0 
    #         att_mask = att_mask.view(1, 1, T, S)
    #         att = att.masked_fill(att_mask == 0, float('-inf'))
    #     else:
    #         att = att.masked_fill(self.mask[:, :, :T, :S] == 0, float('-inf'))
        
    #     att = F.softmax(att, dim=-1)
        
    #     y = att @ v_all  # (B, nh, T, hd)
        
    #     y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
    #     y = self.out_proj(y)
        
    #     return y
    
    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norm)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.attn = CausalSelfAttention(d_model, n_heads)
        
        self.norm1 = RMSNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.SiLU(),
            nn.Linear(4*d_model, d_model)
        )
        
        self.norm2 = RMSNorm(d_model)
    
    def forward(self, x, use_cache=False):
        x = x + self.attn(self.norm1(x), use_cache=use_cache)
        
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def clear_cache(self):
        self.attn.clear_cache()


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=4, n_blocks=4):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_blocks)
        ])
        
        self.norm_out = RMSNorm(d_model)
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # self.apply(self._init_weights)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size

        self.use_cache = False
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens_seq, kv_cache=None, return_cache=False):
        seq_len, batch_size = tokens_seq.shape
        
        x = tokens_seq.transpose(0, 1)
        
        x = self.token_embedding(x)
        
        for block in self.blocks:
            x = block(x, use_cache=self.use_cache)
        
        x = self.norm_out(x)
        
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # (batch, seq_len, vocab_size) -> (seq_len, batch, vocab_size)
        logits = logits.transpose(0, 1)
        
        return logits
    
    def enable_cache(self):
        self.use_cache = True
        
    def disable_cache(self):
        self.use_cache = False
        self.clear_cache()
    
    def clear_cache(self):
        for block in self.blocks:
            block.clear_cache()

################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    probs = F.softmax(logits, dim=0)
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    
    sorted_indices_to_remove = cumulative_probs > p
    
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    
    indices_to_keep = sorted_indices[~sorted_indices_to_remove]
    
    probs_to_keep = sorted_probs[~sorted_indices_to_remove]
    probs_to_keep = probs_to_keep / probs_to_keep.sum()
    
    idx = torch.multinomial(probs_to_keep, 1).item()
    
    next_token = indices_to_keep[idx].item()
    
    return next_token

# def nucleus_sampling(logits, p=0.95):
#     return torch.argmax(logits).item()


def generate_text(model, enc, init_text, max_new_tokens=500, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    END_TOKEN_ID=50256
    if isinstance(model, TransformerModel):
        model.clear_cache()
        model.enable_cache()

    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

            if chosen_token == END_TOKEN_ID:
                break

    if isinstance(model, TransformerModel):
        model.disable_cache()

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-len(annotation_list)])
    # prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)

    return final_text, annotated_text


################################################################################
# 8. Training
################################################################################

def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=300,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a"):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            # current_time = time.time()
            # if current_time >= next_sample_time and enc is not None:
            #     with torch.no_grad():python script.py --block_size 256
            #         print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
            #         text_greedy, ann_greedy = generate_text(
            #             model, enc, prompt, max_new_tokens=500, device=device,
            #             top_p=None,
            #             monosemantic_info=monosemantic_info,
            #             do_monosemantic=(monosemantic_info is not None)
            #         )
            #         print(f" Greedy Sample: {text_greedy}")
            #         print(f" Annotated: {ann_greedy}\n")

            #         print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
            #         text_topp, ann_topp = generate_text(
            #             model, enc, prompt, max_new_tokens=500, device=device,
            #             top_p=0.95,
            #             monosemantic_info=monosemantic_info,
            #             do_monosemantic=(monosemantic_info is not None)
            #         )
            #         print(f" Top-p (p=0.95) Sample: {text_topp}")
            #         print(f" Annotated: {ann_topp}\n")

            #         # third generation => top-p=1.0 => full distribution random sampling
            #         print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
            #         text_topp1, ann_topp1 = generate_text(
            #             model, enc, prompt, max_new_tokens=500, device=device,
            #             top_p=1.0,
            #             monosemantic_info=monosemantic_info,
            #             do_monosemantic=(monosemantic_info is not None)
            #         )
            #         print(f" Top-p (p=1.0) Sample: {text_topp1}")
            #         print(f" Annotated: {ann_topp1}\n")

            #     next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")


        checkpoint_path = os.path.join("model_checkpoints", f"{model_name}_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'global_step': global_step,
        }, checkpoint_path)
        print(f"[{model_name}] Model saved to {checkpoint_path}")
################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = 16
    num_epochs = 1
    learning_rate = 3e-4

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)

            last_idx = 0
            for i in range(0, len(tokens), block_size):
                chunk = tokens[i:i+block_size]
                if len(chunk)>0:
                    tinystories_seqs.append(chunk)
                last_idx = i + block_size

            if last_idx < len(tokens):
                remaining = tokens[last_idx:]
                if len(remaining)>0:
                    tinystories_seqs.append(remaining)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)

                last_idx = 0
                for i in range(0, len(tokens), block_size):
                    chunk = tokens[i:i+block_size]
                    if len(chunk)>0:
                        other_seqs.append(chunk)
                    last_idx = i + block_size

                if last_idx < len(tokens):
                    remaining = tokens[last_idx:]
                    if len(remaining)>0:
                        other_seqs.append(remaining)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    transformer = TransformerModel(
        vocab_size=vocab_size
    ).to(device)

    models = {
      # "kgram_mlp_seq": kgram_model,
        # "lstm_seq": lstm_model,
      "kvcache_transformer": transformer,
    }


    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt  # <--- Pass the user-specified prompt here
        )

        # print(f"\n=== Testing model: {model_name} ===")
        # checkpoint = torch.load('model_checkpoints/kvcache_transformer_epoch_1.pt', map_location=torch.device(args.device_id))
        # model.load_state_dict(checkpoint['model_state_dict'])

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=100, device=device,
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=100, device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=100, device=device,
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
