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
    
# class XavierUniformInitLinear(torch.nn.Linear):
#     # TODO
#     # copied from https://github.com/openvpi/DiffSinger/blob/main/modules/commons/common_layers.py#L29
#     def __init__(
#             self,
#             in_features: int,
#             out_features: int,
#             *args,
#             bias: bool = True,
#             **kwargs
#     ):
#         super().__init__(in_features, out_features, *args, bias=bias, **kwargs)
#         nn.init.xavier_uniform_(self.weight)
#         if bias:
#             nn.init.constant_(self.bias, 0.)

# class TransformerFFNLayer(nn.Module):
#     # TODO I think the design choice here is a bit weird
#     # mostly copied from https://github.com/openvpi/DiffSinger/blob/main/modules/commons/common_layers.py#L120
#     def __init__(self, hidden_size, filter_size, kernel_size=1, dropout=0., act='gelu'):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.dropout = dropout
#         self.act = act
#         filter_size_1 = filter_size
#         if self.act == 'relu':
#             self.act_fn = nn.ReLU()
#         elif self.act == 'gelu':
#             self.act_fn = nn.GELU()
#         elif self.act == 'swish':
#             self.act_fn = nn.SiLU()
#         else:
#             raise ValueError(f'{act} is not a valid activation')
#         self.ffn_1 = nn.Conv1d(hidden_size, filter_size_1, kernel_size, padding=kernel_size // 2)
#         self.ffn_2 = XavierUniformInitLinear(filter_size, hidden_size)

#     def forward(self, x):
#         # x: B x T x C
#         x = self.ffn_1(x.transpose(1, 2)).transpose(1, 2)
#         x = x * self.kernel_size ** -0.5

#         x = self.act_fn(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.ffn_2(x)
#         return x
    
# class EncSALayer(nn.Module):
#     # TODO
#     # copied from https://github.com/openvpi/DiffSinger/blob/main/modules/commons/common_layers.py#L216
#     def __init__(self, c, num_heads, dropout, attention_dropout=0.1,
#                  relu_dropout=0.1, kernel_size=9, act='gelu'):
#         super().__init__()
#         self.dropout = dropout
#         self.layer_norm1 = nn.LayerNorm(c)
#         self.self_attn = nn.MultiheadAttention(c, num_heads, dropout=attention_dropout, bias=False, batch_first=False)
#         self.layer_norm2 = nn.LayerNorm(c)
#         self.ffn = TransformerFFNLayer(c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, act=act)

#     def forward(self, x, encoder_padding_mask=None, **kwargs):
#         layer_norm_training = kwargs.get('layer_norm_training', None)
#         if layer_norm_training is not None:
#             self.layer_norm1.training = layer_norm_training
#             self.layer_norm2.training = layer_norm_training
#         residual = x
#         x = self.layer_norm1(x)

#         x = x.transpose(0, 1)
#         x, _, = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
#         x = x.transpose(0, 1)
        
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = residual + x
#         x = x * (1 - encoder_padding_mask.float())[..., None]

#         residual = x
#         x = self.layer_norm2(x)
#         x = self.ffn(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = residual + x
#         x = x * (1 - encoder_padding_mask.float())[..., None]
#         return x




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
        ###self.positional_embeddings = SinusoidalPositionalEmbedding(embedding_dim=config['embedding_dim'], base=config['sinusoidal_base'])

        head_dim = config['embedding_dim'] // config['num_attention_heads']
        assert config['num_attention_heads'] * head_dim == config['embedding_dim'], f"{config['num_attention_heads']=}, {head_dim=}, {config['embedding_dim']=}, {config['embedding_dim'] % config['num_attention_heads']=}"

        cos, sin = precompute_rotary_embeddings(seq_len=config.max_seq_len, head_dim=head_dim, base=config.rotary_base)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

        # self.layers = nn.ModuleList([
        #     EncSALayer( # TODO this class is a mess
        #         self.hidden_size, self.dropout,
        #         kernel_size=ffn_kernel_size, act=ffn_act,
        #         num_heads=num_heads, rotary_embed=rotary_embed
        #     )
        #     for _ in range(self.num_layers)
        # ])

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_blocks)])

        ###self.layer_norm = nn.LayerNorm(config['embedding_dim'])

    
    def forward(self, token_ids, cond, ph_padding_mask):
        """
        B is batch size
        P is phoneme sequence length
        `token_ids` has shape (B, P) and contains the token ids corresponding to the phonemes
        `cond` has shape (B, P, embedding_dim) -- embeddings to condition on -- in practice, this will be the phoneme duration embeddings created in the music score encoder forward pass
        """
        tok_embs = self.token_embeddings(token_ids) # (B, P, embedding_dim)
        B, P, embedding_dim = token_ids.shape
        ###pos_embs = self.positional_embeddings(seq_len=P, device=tok_embs.dvice) # not using RoPE
        cos_sin = self.cos, self.sin

        x = tok_embs * math.sqrt(embedding_dim) # scale embeddings by sqrt(d)
        x = x + cond### + pos_embs
        x = F.dropout(x, p=self.dropout, training=self.training)
        # TODO attention and stuff? everything you did so far takes you roughly up to here https://github.com/openvpi/DiffSinger/blob/main/modules/fastspeech/tts_modules.py#L408

        # x = x * torch.logical_not(ph_padding_mask)
        # for layer in self.layers:
        #     x = layer(x, encoder_padding_mask=ph_padding_mask, attn_mask=attn_mask) * torch.logical_not(ph_padding_mask) # TODO they don't even have an attn_mask arg, this is such a mess
        # x = self.layer_norm(x) * torch.logical_not(ph_padding_mask)
        # return x
        attn_mask = torch.logical_not(ph_padding_mask)
        x = rmsnorm(x)
        for block in self.blocks:
            x = block(x=x, cos_sin=cos_sin, attn_mask=attn_mask)
        #####x = F.layer_norm(x, [x.shape[-1]]) * scale + shift
        x = rmsnorm(x)
        return x # (B, P, embedding_dim)

class MusicScoreEncoder(nn.module):

    def __init__(self, config):
        super().__init__()
        #self.txt_embed = nn.Embedding(config['vocab_size'], config['embedding_dim'], config['pad_token_id'])
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
        """

        ######txt_embed = self.txt_embed(txt_tokens) # (B, P, embedding_dim) # going to remove this since we'll use PhonemeTextEncoder which has the embedding layer inside of it
        
        dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float() # (B, P)
        dur_embed = self.dur_embed(dur[:, :, None]) # (B, P, embedding_dim) -- (B, P, 1) x (1, embedding_dim) ### each row is the same embedding weights scaled by the duration (the same bias is added to each row)

        phoneme_text_embeddings = self.phoneme_text_encoder(token_ids=txt_tokens, cond=dur_embed, ph_padding_mask=ph_padding_mask) # (B. P, embedding_dim)

        mel2ph_ = mel2ph[..., None].repeat([1, 1, phoneme_text_embeddings.shape[-1]]) # (B, T, embedding_dim)

        # Typically T >> P. For each mel frame we extract the phoneme embedding corresponding to index stored in mel2ph
        condition = torch.gather(input=phoneme_text_embeddings, dim=1, index=mel2ph_) # (B, T, embedding_dim) -- note: probably want padding token to be index 0 or something to make it a valid index to avoid an error here, can deal with ignoring padding embedding terms later

        f0_mel = (1 + f0 / 700).log() # (B, T)
        pitch_embed = self.pitch_embed(f0_mel[:, :, None]) # (B, T, embedding_dim)
        condition += pitch_embed

        return condition # (B, T, embedding_dim)

        # # TODO below here -----------------
        # # next, they do some transformer-based encoding of the text with positional info and padding taken into account, this makes sense at a high level
        # # seems like i should put self.txt_embed inside the encoder and just call it self.txt_encoder or something, seems simpler, feed in padding mask too
        # encoder_out = self.encoder(txt_embed, extra_embed, txt_tokens == 0) # takes dur_emb as input, maybe rename some stuff
        # # see https://github.com/openvpi/DiffSinger/blob/main/modules/fastspeech/tts_modules.py#L407 for phoneme text encoder

        # #encoder_out = F.pad(encoder_out, [0, 0, 1, 0]) # I don't need this because i 0-index the mel2ph
        # mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        # # continue around here https://github.com/openvpi/DiffSinger/blob/main/modules/fastspeech/acoustic_encoder.py#L100



class AuxiliaryDeocder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config['num_layers'] # they use 6 apparently, see e.g. Section 4.2 https://arxiv.org/abs/1905.09263