"""
This script parses all of the TextGrid files in PopCS to build a phoneme vocabulary which includes special
tokens. This vocabulary is saved in `DiffSinger/binarized_data/vocab.json`. Additionally, this builds train,
val, and test splits where the val and test splits are defined in the config. The splits are stored in
`DiffSinger/binarized_data/splits.json`.

Usage:
    uv run preprocess_data.py --config /home/phil/DiffSinger/data/configs/binarize-PopCS.toml

"""

import json
import random
from pathlib import Path
import argparse
import textgrid
import tomllib


def build_phoneme_vocab(raw_data_dir: str, save_dir: str) -> dict[str, int]:
    """
    `raw_data_dir` is expected to be a directory containing subdirectories which contain TextGrid files, this function
    iterates through all of these TextGrid files and collects the phonemes from Tier 2 of the TextGrid. 

    Output is a dictionary with string token keys (corresponding to phonemes or special tokens) and integer values. 
    A .json file of the collected vocabulary is saved in `save_dir`

    Example directory structure:

        popcs                   <-- raw_data_dir (for PopCS dataset)
            popcs-Bad           <-- example subdirectory
                0000.TextGrid   <-- first TextGrid file we parse
                0000.txt       
                0000_ph.txt
                0000_wf0.wav
                0001.TextGrid   <-- second TextGrid file we parse
                ...

    """
    special_tokens = ["<PAD>", "<BOS>", "<EOS>", "SP"]
    raw_data_dir = Path(raw_data_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vocab = set() # store phonemes
    for song_dir in sorted(raw_data_dir.iterdir()):
        for text_grid_path in sorted(song_dir.glob("*.TextGrid")):
            text_grid = textgrid.TextGrid.fromFile(str(text_grid_path))
            tier = text_grid.tiers[1] # we use index 1 because we want the second tier because that it what other people used
            for interval in tier:
                phoneme = interval.mark.strip()
                if len(phoneme) > 0:
                    vocab.add(phoneme)

    vocab = special_tokens + sorted(vocab)
    vocab = {token : i for i, token in enumerate(vocab)}
    save_path = save_dir / "vocab.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Phoneme vocabulary: {len(vocab)} tokens (including special tokens: {special_tokens})")
    print(f"Tokens: {vocab}")
    print(f"Saved to {save_path}")

    return vocab


def build_splits(raw_data_dir: str, save_dir: str, val_songs: list[str], test_songs: list[str]) -> dict[str, list[str]]:
    """
    Give lists of PopCS song names for the val and test sets, for example in DiffSinger they use
        test_songs = ["popcs-说散就散", "popcs-隐形的翅膀"]
        (in DiffSinger they made the val set the same as the test set, looks like it might be unintentional based on this indexing https://github.com/MoonInTheRiver/DiffSinger/blob/ce7789f1427ddcdec647b3ab2bf2d1b12134e51e/data_gen/tts/base_binarizer.py#L65)
    The lists get saved to a .json file stored in `save_dir`
    """
    raw_data_dir = Path(raw_data_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    test_songs  = set(test_songs)
    val_songs = set(val_songs)

    splits = {"test": [], "val": [], "train": []}

    for song_dir in sorted(raw_data_dir.iterdir()):
        song_name = song_dir.name
        for text_grid_path in sorted(song_dir.glob("*.TextGrid")):
            item_name = f"{song_dir.name}-{text_grid_path.stem}"

            if song_name in test_songs:
                splits["test"].append(item_name)
            elif song_name in val_songs:
                splits["val"].append(item_name)
            else:
                splits["train"].append(item_name)

    save_path = save_dir / "splits.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    for split, names in splits.items():
        print(f"{split:>5}: {len(names)} utterances saved in {save_path}")

    return splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help=".toml file")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    config = config['data']

    vocab = build_phoneme_vocab(
        raw_data_dir=config['raw_data_dir'], 
        save_dir=config['save_dir']
        )
    
    splits = build_splits(
        raw_data_dir=config['raw_data_dir'], 
        save_dir=config['save_dir'], 
        val_songs=config['val_songs'],
        test_songs=config['test_songs']
        )

