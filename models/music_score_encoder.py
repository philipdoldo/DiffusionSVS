import torch
import torch.nn as nn


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


class MusicScoreEncoder(nn.module):

    def __init__(self, config):
        super().__init__()
        self.txt_embed = nn.Embedding(config['vocab_size'], config['embedding_dim'], config['pad_token_id'])
        self.dur_embed = nn.Linear(1, config['embedding_dim'])
    

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

        txt_embed = self.txt_embed(txt_tokens) # (B, P, embedding_dim)
        dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float() # (B, P)
        dur_embed = self.dur_embed(dur[:, :, None]) # (B, P, embedding_dim) -- (B, P, 1) x (1, embedding_dim) ### each row is the same embedding weights scaled by the duration (the same bias is added to each row)

        # TODO below here -----------------
        # next, they do some transformer-based encoding of the text with positional info and padding taken into account, this makes sense at a high level
        # seems like i should put self.txt_embed inside the encoder and just call it self.txt_encoder or something, seems simpler, feed in padding mask too
        encoder_out = self.encoder(txt_embed, extra_embed, txt_tokens == 0) # takes dur_emb as input, maybe rename some stuff
        # see https://github.com/openvpi/DiffSinger/blob/main/modules/fastspeech/tts_modules.py#L407 for phoneme text encoder

        #encoder_out = F.pad(encoder_out, [0, 0, 1, 0]) # I don't need this because i 0-index the mel2ph
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        # continue around here https://github.com/openvpi/DiffSinger/blob/main/modules/fastspeech/acoustic_encoder.py#L100