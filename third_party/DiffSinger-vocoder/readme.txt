I downloaded the vocoder weights from https://github.com/MoonInTheRiver/DiffSinger/blob/master/docs/README-SVS-popcs.md which has a Google Drive link https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0109_hifigan_bigpopcs_hop128.zip
The directory structure is:
third_party/DiffSinger-vocoder/
├── code/
├── config.yaml
├── model_ckpt_steps_280000.ckpt
├── readme.txt
└── 0109_hifigan_bigpopcs_hop128.zip


the `code` directory is the DiffSinger repo git cloned using e.g. `git clone --depth=1 https://github.com/MoonInTheRiver/DiffSinger /home/phil/DiffusionSVS/third_party/DiffSinger-vocoder/code`

Edits to DiffSinger source code:
- I ran `sed -i 's/from scipy.signal import kaiser/from scipy.signal.windows import kaiser/' third_party/DiffSinger-vocoder/code/modules/parallel_wavegan/layers/pqmf.py`
This was to get around the error:

        (base) phil@DESKTOP-5AGLK92:~/DiffusionSVS$ uv run vocoder.py
        Traceback (most recent call last):
        File "/home/phil/DiffusionSVS/vocoder.py", line 9, in <module>
            from modules.hifigan.hifigan import HifiGanGenerator
        File "/home/phil/DiffusionSVS/third_party/DiffSinger-vocoder/code/modules/hifigan/hifigan.py", line 7, in <module>
            from modules.parallel_wavegan.layers import UpsampleNetwork, ConvInUpsampleNetwork
        File "/home/phil/DiffusionSVS/third_party/DiffSinger-vocoder/code/modules/parallel_wavegan/layers/__init__.py", line 2, in <module>
            from .pqmf import *  # NOQA
            ^^^^^^^^^^^^^^^^^^^
        File "/home/phil/DiffusionSVS/third_party/DiffSinger-vocoder/code/modules/parallel_wavegan/layers/pqmf.py", line 12, in <module>
            from scipy.signal import kaiser
        ImportError: cannot import name 'kaiser' from 'scipy.signal' (/home/phil/DiffusionSVS/.venv/lib/python3.12/site-packages/scipy/signal/__init__.py). Did you mean: 'kaiserord'?

Claude said:
scipy.signal.kaiser was removed in newer scipy — it's now at scipy.signal.windows.kaiser. Patch the file:
`sed -i 's/from scipy.signal import kaiser/from scipy.signal.windows import kaiser/' third_party/DiffSinger-vocoder/code/modules/parallel_wavegan/layers/pqmf.py`