"""Dataset-specific phone folding maps."""

from __future__ import annotations

ENGLISH_DATASET_PHONE_FOLD_MAPS: dict[str, dict[str, str]] = {
    "AlexFloarea-AI-SVS": {
        "M": "m",
        "N": "n",
        "exh": "hh",
        "i": "iy",
        "oh": "ow",
        "q": "cl",
        "vf": "v",
    },
    "Alex_Floarea_EN_Public_Corpus": {
        "ctrash": "cl",
        "fr/r": "er",
        "nn": "n",
        "q": "cl",
        "vf": "v",
    },
    "Project-AIdol-Public-English-Dataset": {
        "GS": "cl",
        "en": "n",
        "vf": "v",
    },
    "tiger_en": {
        "ctrash": "cl",
        "q": "cl",
        "vf": "v",
    },
}


JAPANESE_DATASET_PHONE_FOLD_MAPS: dict[str, dict[str, str]] = {
    "Amaboshi_CipherDB": {
        "Edge": "SP",
        "GlottalStop": "cl",
        "O": "o",
        "fy": "hy",
    },
    "Kurotake_Kouga_AI_Song": {
        "fy": "hy",
        "vy": "v",
    },
    "PJS_corpus_ver1.1": {
        "O": "o",
    },
    "enunu_kodoku_database_20220807-2": {
        "Edge": "SP",
        "GlottalStop": "cl",
    },
    "nit070_db": {
        "q": "cl",
    },
    "ritsu": {
        "A": "a",
        "Edge": "SP",
        "GlottalStop": "cl",
        "O": "o",
        "fy": "hy",
    },
    "tiger_jp": {
        "vf": "v",
    },
    "jvs_ver1_mon": {
        "c": "ts",
    },
}


MANDARIN_PHONE_FOLD_MAP: dict[str, str] = {
    "<AP>": "AP",
    "<SP>": "SP",
    "AP": "AP",
    "SP": "SP",
    "sil": "SP",
    "sp": "SP",
    "sp1": "SP",
    "iu": "iou",
    "ui": "uei",
    "un": "uen",
}


MANDARIN_ZERO_INITIAL_PAIR_FOLD_MAP: dict[tuple[str, str], str] = {
    ("w", "a"): "ua",
    ("w", "ai"): "uai",
    ("w", "an"): "uan",
    ("w", "ang"): "uang",
    ("w", "ei"): "uei",
    ("w", "en"): "uen",
    ("w", "o"): "uo",
    ("w", "u"): "u",
    ("y", "a"): "ia",
    ("y", "an"): "ian",
    ("y", "ang"): "iang",
    ("y", "ao"): "iao",
    ("y", "e"): "ie",
    ("y", "i"): "i",
    ("y", "in"): "in",
    ("y", "ing"): "ing",
    ("y", "ong"): "iong",
    ("y", "ou"): "iou",
    ("y", "u"): "v",
    ("y", "v"): "v",
    ("y", "van"): "van",
    ("y", "ve"): "ve",
    ("y", "vn"): "vn",
}


# GTSinger is weird, man
GTSINGER_JAPANESE_PHONE_FOLD_MAP: dict[str, str] = {
    "<AP>": "AP",
    "<SP>": "SP",
    "aː": "a",
    "eː": "e",
    "iː": "i",
    "oː": "o",
    "bʲ": "by",
    "c": "ts",
    "mʲ": "my",
    "ɾ": "r",
    "ɾʲ": "ry",
    "ɲ": "ny",
    "ç": "hy",
    "ɕ": "sh",
    "tɕ": "ch",
    "ɡ": "g",
    "ɟ": "gy",
    "dz": "z",
    "dʑ": "j",
    "ʑ": "j",
    "ɸ": "f",
    "ɴ": "N",
    "ɴː": "N",
    "nː": "N",
    "ɰ̃": "w",
    "ʔ": "cl",
    "pː": "cl",
    "tː": "cl",
}


__all__ = [
    "ENGLISH_DATASET_PHONE_FOLD_MAPS",
    "GTSINGER_JAPANESE_PHONE_FOLD_MAP",
    "JAPANESE_DATASET_PHONE_FOLD_MAPS",
    "MANDARIN_PHONE_FOLD_MAP",
    "MANDARIN_ZERO_INITIAL_PAIR_FOLD_MAP",
]
