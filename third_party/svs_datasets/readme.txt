`third_party/svs_datasets/code` contains the svs_datasets repo (https://github.com/Neolyre/svs_datasets) with a small edit made (as of 08/02/2026)


`/svs_datasets/preprocessing/adapters/m4singer.py` was edited to change 
`notes = tuple(str(note) for note in item["notes"])` to `notes = tuple(int(note) for note in item["notes"])`
inside the `_note_intervals_from_item` function. This was to avoid an error.