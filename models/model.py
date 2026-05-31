"""
Some files worth referencing if you want a more faithful replication. I chose to ignore some details for simplicity.
    OpenVPI's music score encoder: https://github.com/openvpi/DiffSinger/blob/main/modules/fastspeech/acoustic_encoder.py#L14
    OpenVPI's phoneme text encoder (used in music score encoder): https://github.com/openvpi/DiffSinger/blob/main/modules/fastspeech/tts_modules.py#L353
    Some modules they use for feedforward transformer layer, I just ignored these to keep things simple for now (what I do might be better anyway, e.g. I use RoPE)
        https://github.com/openvpi/DiffSinger/blob/main/modules/commons/common_layers.py#L29
        https://github.com/openvpi/DiffSinger/blob/main/modules/commons/common_layers.py#L120
        https://github.com/openvpi/DiffSinger/blob/main/modules/commons/common_layers.py#L216
    OpenVPI's auxiliary decoder: https://github.com/openvpi/DiffSinger/blob/main/modules/aux_decoder/convnext.py -- I ignored this for now and did my own implementation and to keep things simple
    OpenVPI's wavenet: https://github.com/openvpi/DiffSinger/blob/main/modules/backbones/wavenet.py
    DiffSinger's wavenet: https://github.com/MoonInTheRiver/DiffSinger/blob/ce7789f1427ddcdec647b3ab2bf2d1b12134e51e/usr/diff/net.py#L81    

    In the original code, they 1-index mel2ph but I choose to 0-index mel2ph which makes things simpler 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def mel2ph_to_dur(mel2ph, P, max_dur=None):
    """
    `mel2ph` has shape (B, T)
    `P` is an integer corresponding to how long the phoneme sequence of `txt_tokens` is
    This function returns a shape (B, P) tensor `dur` where index i of the phoneme sequence is the number of mel frames that
    position i of `txt_tokens` lasted for, so basically just a tensor of the durations of each phoneme in `txt_tokens`. 
    
    The durations tensor `dur` is used in the encoder to create a duration embedding by passing `dur` into a linear layer,
    the duration embedding is provided to the phoneme text encoder as conditioning.
    """
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, P).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur # shape (B, P)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim=256, base=10000):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding embedding dimension {embedding_dim=} must be a multiple of 2")
        self.embedding_dim = embedding_dim
        self.base = base # e.g. 10000
    
    def forward(self, seq_len: int, device):
        half_dim = self.embedding_dim // 2
        i = torch.arange(half_dim, device=device) # shape (embedding_dim//2)
        freqs = 1/self.base ** (i / (half_dim - 1)) # shape (embedding_dim//2) # e.g. frequencies ranging from 1 down to 1/10000
        positions = torch.arange(seq_len, device=device) # shape (seq_len)
        angles = positions[:, None] * freqs[None, :] # shape (seq_len, embedding_dim//2)
        return torch.cat([angles.sin(), angles.cos()], dim=-1) # shape (seq_len, embedding_dim)

def rmsnorm(x):
    """
    `x` has shape (B, L, d) and the RMS (which is just 2-norm scaled by 1/sqrt(d) in R^d) is computed for every vector of channels
    No learnable parameters. I want to try rmsnorm since I used it in language models. 
    """
    rms = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-8).sqrt() # shape (B, L, 1)
    return (x / rms)

def precompute_rotary_embeddings(seq_len, head_dim, base=10000):
    # stride the channels
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    # stride the time steps
    t = torch.arange(seq_len, dtype=torch.float32)
    # calculate the rotation frequencies at each (time, channel) pair
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    #cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
    cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
    return cos, sin

def apply_rotary_emb(x, cos, sin):
    """
    `cos` and `sin` each have shape [1, max_seq_len, 1, head_dim // 2]
    `x` has shape [batch_size, seq_len, num_heads, head_dim]
    """
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2 # head_dim // 2

    # Truncate `cos` and `sin` to the input sequence length
    input_seq_len = x.shape[1]
    cos = cos[:, :input_seq_len, :, :]
    sin = sin[:, :input_seq_len, :, :]

    x1, x2 = x[..., :d], x[..., d:] # split up head dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims (they rotate clockwise, arbitrary choice that I am copying)
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], dim=3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class BidirectionalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.Wq = nn.Linear(config['embedding_dim'], config['embedding_dim'], bias=False)
        self.Wk = nn.Linear(config['embedding_dim'], config['embedding_dim'], bias=False)
        self.Wv = nn.Linear(config['embedding_dim'], config['embedding_dim'], bias=False)
        self.Wo = nn.Linear(config['embedding_dim'], config['embedding_dim'], bias=False)

        self.num_heads = config['num_attention_heads']
        self.embed_dim = config['embedding_dim']

    def forward(self, x, cos_sin, attn_mask):
        """
        `x` has shape (batch_size, seq_len, embed_dim)
        `attn_mask` has shape (batch_size, seq_len) and is True for entries that should take part in attention and False otherwise
        """
        batch_size, seq_len, embed_dim = x.shape
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x) # each shas shape (batch_size, seq_len, embed_dim)

        head_dim = embed_dim // self.num_heads
        assert self.num_heads * head_dim == embed_dim, f"{self.num_heads=}, {head_dim=}, {embed_dim=}"

        q = q.view(batch_size, seq_len, self.num_heads, head_dim) # (batch_size, seq_len, num_heads, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim) # (batch_size, seq_len, num_heads, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim) # (batch_size, seq_len, num_heads, head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = rmsnorm(q), rmsnorm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, e.g. (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)

        # att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.shape[-1])) # (batch_size, num_heads, seq_len, seq_len)
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.Wo(y)

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim=None, output_dim=None, bias=False):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * input_dim
        output_dim = output_dim if output_dim is not None else input_dim

        self.W1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x):
        """
        `x` has shape (B, d) where B is batch size and d is embedding dimension
        """
        x = self.W1(x)
        x = F.silu(x) # TODO pass in different activation function options from config
        x = self.W2(x)
        return x
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = BidirectionalSelfAttention(config)
        self.mlp = MLP(input_dim=config.embed_dim, hidden_dim=4*config.embed_dim, output_dim=config.embed_dim)

    def forward(self, x, cos_sin, attn_mask):
        """
        `x` has shape (B, P, embedding_dim) where B is batch size and P is the phoneme sequence length
            this is basically a tensor containing sequences of phoneme/token embeddings 
        """
        x = x + self.attn(x=rmsnorm(x), cos_sin=cos_sin, attn_mask=attn_mask)
        x = x + self.mlp(rmsnorm(x))
        return x

class PhonemeTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embbeddings = nn.Embedding(config['vocab_size'], config['embedding_dim'], config['pad_token_id'])

        head_dim = config['embedding_dim'] // config['num_attention_heads']
        assert config['num_attention_heads'] * head_dim == config['embedding_dim'], f"{config['num_attention_heads']=}, {head_dim=}, {config['embedding_dim']=}, {config['embedding_dim'] % config['num_attention_heads']=}"

        cos, sin = precompute_rotary_embeddings(seq_len=config.max_seq_len, head_dim=head_dim, base=config.rotary_base)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config['num_blocks'])])
    
    def forward(self, token_ids, cond, ph_padding_mask):
        """
        B is batch size
        P is phoneme sequence length
        `token_ids` has shape (B, P) and contains the token ids corresponding to the phonemes
        `cond` has shape (B, P, embedding_dim) -- embeddings to condition on -- in practice, this will be the phoneme duration embeddings created in the music score encoder forward pass
        """
        tok_embs = self.token_embeddings(token_ids) # (B, P, embedding_dim)
        B, P, embedding_dim = token_ids.shape
        cos_sin = self.cos, self.sin

        x = tok_embs * math.sqrt(embedding_dim) # scale embeddings by sqrt(d)
        x = x + cond # I'm using RoPE, but if not using RoPE you'd also add sinusoidal positional embeddings here if following the original implementation
        x = F.dropout(x, p=self.dropout, training=self.training)

        attn_mask = torch.logical_not(ph_padding_mask) # (B, P)
        x = rmsnorm(x)
        for block in self.blocks:
            x = block(x=x, cos_sin=cos_sin, attn_mask=attn_mask)
        x = rmsnorm(x)
        return x # (B, P, embedding_dim)

class MusicScoreEncoder(nn.module):

    def __init__(self, config):
        super().__init__()
        self.phoneme_text_encoder = PhonemeTextEncoder(config)
        self.dur_embed = nn.Linear(1, config['embedding_dim'])
        self.pitch_embed = nn.Linear(1, config['embedding_dim'])
    

    def forward(self, txt_tokens, mel2ph, f0, uv, ph_padding_mask, mel_padding_mask):
        """
        `txt_tokens` has shape (B, P)  --  where B is batch size and P is the number of phonemes in the sequence
            contains sequences of phoneme token ids (corresponding to the phonemes used in an audio file)
        `mel2ph` has shape (B, T)  --  where T is the number of mel frames (when constructing the mel-spectrogram, time was discretized into T mel frames)
            `mel2ph` on a given mel frame contains the `txt_token` index corresponding to the phoneme used on in the audio, it is important to note that
            this is not the token id but the index into `txt_tokens`, this will allow for the same token id to potentially receive different positional
            information if the same token is used multiple times in `txt_tokens`
        `f0` has shape (B, T)
            contains the fundamental frequency during each mel frame (interpolation was used to smooth across unvoiced segments, see data preprocessing/binarization code)
        `uv` has shape (B, T)
            boolean mask which is True when the audio was unvoiced, corresponds to where the preinterpolated f0 was zero. False otherwise.
            TODO `uv` appears to be unused in training? possibly used in validation?
        """
        dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float() # (B, P)
        dur_embed = self.dur_embed(dur[:, :, None]) # (B, P, embedding_dim) -- (B, P, 1) x (1, embedding_dim) ### each row is the same embedding weights scaled by the duration (the same bias is added to each row)

        phoneme_text_embeddings = self.phoneme_text_encoder(token_ids=txt_tokens, cond=dur_embed, ph_padding_mask=ph_padding_mask) # (B. P, embedding_dim)

        mel2ph_ = mel2ph[..., None].repeat([1, 1, phoneme_text_embeddings.shape[-1]]) # (B, T, embedding_dim)
        # Typically T >> P. For each mel frame we extract the phoneme embedding corresponding to index stored in mel2ph
        condition = torch.gather(input=phoneme_text_embeddings, dim=1, index=mel2ph_) # (B, T, embedding_dim) -- note: probably want padding token to be index 0 or something to make it a valid index to avoid an error here, can deal with ignoring padding embedding terms later

        f0_mel = (1 + f0 / 700).log() # (B, T) # TODO, paper says f0 is standardied to mean 0 and unit variance, but idk where this happens
        pitch_embed = self.pitch_embed(f0_mel[:, :, None]) # (B, T, embedding_dim)
        condition += pitch_embed

        return condition # (B, T, embedding_dim)

class AuxiliaryDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_blocks = config['num_blocks'] # they use 6 apparently, see e.g. Section 4.2 https://arxiv.org/abs/1905.09263
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['num_blocks'])])
        self.W = nn.Linear(config['embedding_dim'], config['mel_bins']) # e.g. project from 256 to 80

        head_dim = config['embedding_dim'] // config['num_attention_heads']
        assert config['num_attention_heads'] * head_dim == config['embedding_dim'], f"{config['num_attention_heads']=}, {head_dim=}, {config['embedding_dim']=}, {config['embedding_dim'] % config['num_attention_heads']=}"

        cos, sin = precompute_rotary_embeddings(seq_len=config.max_seq_len, head_dim=head_dim, base=config.rotary_base)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, x, mel_padding_mask):
        """
        `x` has shape (B, T, d) where B is batch size, T is mel frames, and d is embedding dimension. Intended to be the output of
            the music score encoder, the decoder is meant to transform it into a batch of mel-spectrograms with shape (B, T, M)
            where M is the number of mel bins (discretizes the frequency, usually use M=80)
        `mel_padding_mask` has shape (B, T) and is True if the mel frame index (for a given batch index) corresponds to a padding value, False otherwise
        """
        cos_sin = self.cos, self.sin
        attn_mask = torch.logical_not(mel_padding_mask)
        x = rmsnorm(x) # (B, T, d)
        for block in self.blocks:
            x = block(x=x, cos_sin=cos_sin, attn_mask=attn_mask)
        x = self.W(x)
        return x # (B, T, M)


###########################################
# TODO
# TODO, rewrite the wavenet and residual block below, uv is only used during validation? how do they predict k again? need pad collate function, need training loop, loss functions, need to train enc/dec and denoiser separately, handle freezing weights, etc. need vocoder if you're gonna validate sounds produced from test set gen'd spectrograms
class ResidualBlock(nn.Module):
    def __init__(self, embedding_dim=256, dilation=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dilated_conv = nn.Conv1d(in_channels=embedding_dim, out_channels=2*embedding_dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.time_emb_projection = nn.Linear(embedding_dim, embedding_dim)
        self.cond_projection = nn.Conv1d(in_channels=embedding_dim, out_channels=2*embedding_dim, kernel_size=1)
        self.output_projection = nn.Conv1d(in_channels=embedding_dim, out_channels=2*embedding_dim, kernel_size=1)

    def forward(self, x, cond, time_emb):
        """
        `x` has shape (B, d, T) ###(B, T, M) maybe???
        `cond` has shape (B, d, T) and is the output from the music score encoder
        `time_emb` has shape (B, d) and is a batch of diffusion time steps
        """
        time_emb = self.time_emb_projection(time_emb).unsqueeze(-1) # (B, d, 1)
        cond = self.cond_projection(cond) # (B, 2*d, T)
        y = x + time_emb # (B, d, T)

        y = self.dilated_conv(y) + cond # (B, 2*d, T)

        gate, filter = torch.split(y, [self.embedding_dim, self.embedding_dim], dim=1) # ((B, d, T), (B, d, T))
        y = torch.sigmoid(gate) * torch.tanh(filter) # (B, d, T)

        y = self.output_projection(y) # (B, 2*d, T)

        residual, skip = torch.split(y, [self.embedding_dim, self.embedding_dim], dim=1) # ((B, d, T), (B, d, T))
        return (x + residual) / math.sqrt(2.0), skip # ((B, d, T), (B, d, T))

class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)

class WaveNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_mel_bins = config['num_mel_bins']
        self.input_projection = Conv1d(in_channels=config['num_mel_bins'], out_channels=config['embedding_dim'], kernel_size=1)
        self.time_embedding = SinusoidalPositionalEmbedding(embedding_dim=config['embedding_dim'], base=config['sinusoidal_base'])
        self.mlp = nn.Sequential(
            nn.Linear(config['embedding_dim'], config['embedding_dim']*4),
            nn.Mish(),
            nn.Linear(config['embedding_dim']*4, config['embedding_dim'])
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                embedding_dim=config['embedding_dim'],
                residual_channels=config['embedding_dim'],
                dilation=2**(i % config['dilation_cycle_length'])
            )
            for i in range(config['num_wavenet_layers'])
        ])
        self.skip_projection = Conv1d(in_channels=config['embedding_dim'], out_channels=config['embedding_dim'], kernel_size=1)
        self.output_projection = Conv1d(in_channels=config['embedding_dim'], out_channels=config['num_mel_bins'], kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, mel, t, cond):
        """
        `mel` has shape (B, M, T) -- the mel spectrogram
            B is batch size
            M is number of mel bins
            T is number of mel frames ### TODO note: in collate function or binarization code you'll need to transpose from (T, M) to (M, T) and maybe unsqueeze a 1
        `t` has shape (B, 1) which is a batch of diffusion time steps
        `cond` has shape (B, T, d)
            d is embedding dimension
        """
        B, M, T = mel.shape 
        if M != self.num_mel_bins:
            raise ValueError(f"{mel.shape=}, {self.num_mel_bins=}")
        if cond.shape[-1] != T:
            raise ValueError(f"{cond.shape=}, {T=}")
        cond.transpose(-2, -1) # (B, d, T)
        x = self.input_projection(mel)  # (B, d, T)

        x = F.relu(x) # (B, d, T)
        time_emb = self.time_embedding(t) # (B, d)
        time_emb = self.mlp(time_emb) # (B, d)
        skip_connections = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x=x, cond=cond, time_emb=time_emb) # ((B, d, T), (B, d, T))
            skip_connections.append(skip_connection)

        x = torch.sum(torch.stack(skip_connections), dim=0) / math.sqrt(len(self.residual_layers)) # (B, d, T) -- stack results in (num_layers, B, d, T) and sum over dim 0 results in (B, d, T)
        x = self.skip_projection(x) # (B, d, T)
        x = F.relu(x) # (B, d, T)
        x = self.output_projection(x) # (B, M, T)
        return x