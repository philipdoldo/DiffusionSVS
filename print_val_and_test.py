import sys
sys.path.insert(0, "third_party/svs_datasets/code")
from third_party.svs_datasets.code import  CanonicalExample, Dataset, Interval, NoteInterval, load_dataset
from binarize import get_dataset_paths
import toml
import argparse

def get_specific_examples(examples, dirs):
        specific_examples = []
        for ex in examples:
            for directory in dirs:
                if directory in ex.audio_path:
                    assert ex not in specific_examples, f"{specific_examples=}, {ex=}"
                    specific_examples.append(ex)
        s = ""
        for x in specific_examples:
            s += f'"{x.audio_path}",\n'
            #s += f'["{x.source_dataset}", "{x.utterance_id}"],\n' # this is outdated, never use this because utterance_id isn't unique per dataset
        return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help=".toml file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = toml.load(f)

    dataset_paths = get_dataset_paths(config['data'])

    datasets = [Dataset.POPCS, Dataset.GTSINGER_ENGLISH, Dataset.GTSINGER_JAPANESE]

    base_dir = "/home/phil/DiffusionSVS/data"

    test_string = ""
    val_string = ""

    for dataset in datasets:
        path = dataset_paths[dataset]
        examples = load_dataset(dataset, path)
        print(f"    DATASET {dataset} LOADED: {len(examples)=}")

        assert len(examples) > 0

        if dataset == Dataset.GTSINGER_ENGLISH:
            test_dirs = [f"English-005/English/EN-Tenor-1/Vibrato/You Belong With Me/Vibrato_Group", "gtsinger/English-005/English/EN-Alto-2/Pharyngeal/You Raise Me Up/Pharyngeal_Group"]#"/home/phil/DiffusionSVS/data/gtsinger/English-005/English/EN-Alto-2/Pharyngeal/Young And Beautiful/Pharyngeal_Group"]
            val_dirs = ["gtsinger/English-005/English/EN-Alto-1/Glissando/trouble is a friend/Glissando_Group", "gtsinger/English-005/English/EN-Alto-2/Breathy/My Love/Breathy_Group"]# "/home/phil/DiffusionSVS/data/gtsinger/English-005/English/EN-Alto-2/Breathy/Safe and Sound/Breathy_Group"]
        elif dataset == Dataset.GTSINGER_JAPANESE:
            test_dirs = ["gtsinger/Japanese-008/Japanese/JA-Tenor-1/Vibrato/πüòπéôπü╗πéÜ/Vibrato_Group", "gtsinger/Japanese-008/Japanese/JA-Tenor-1/Pharyngeal/πüòπéôπü╗πéÜ/Pharyngeal_Group"]
            val_dirs = ["gtsinger/Japanese-008/Japanese/JA-Soprano-1/Mixed_Voice_and_Falsetto/Φíîπüïπü¬πüäπüªπéÖ/Falsetto_Group", "gtsinger/Japanese-008/Japanese/JA-Soprano-1/Glissando/Σ╜òΦë▓πüªπéÖπééπü¬πüäΦè▒/Glissando_Group"]
        elif dataset == Dataset.POPCS:
            test_dirs = ["popcs/popcs-说散就散", "popcs/popcs-隐形的翅膀"]
            val_dirs = ["popcs/popcs-夏天的风", "popcs/popcs-稻香"]
        else:
            test_dirs = []
            val_dirs = []
            print("not handled")
            raise ValueError(f"check inputs")

        test_string += get_specific_examples(examples=examples, dirs=test_dirs)
        val_string += get_specific_examples(examples=examples, dirs=val_dirs)

    print("TEST SET:")
    print(test_string)
    print("\n\n\n")
    print("VAL SET:")
    print(val_string)
