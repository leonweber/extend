# Extend, don't rebuild: Phrasing conditional graph modification as autoregressive sequence labelling
Code for the paper *Extend, don't rebuild: Phrasing conditional graph modification as autoregressive sequence labelling, EMNLP 2021*

## Installation
```bash
pip install -r requirements.txt
```
For GPU acceleration, please install a suitable version of [PyTorch](https://pytorch.org/get-started/locally/).

## Train
```bash
python -m extend.train dataset=scene dataset.name=crowdsourced model.num_epochs=100

 $ ls outputs/2021-11-01/17-12-02/
'epoch=19-step=8719.ckpt'   train.log   wandb
```

## Evaluate Biomedical Event Graph Modification
```bash
python -m extend.eval.predict_extend dataset=bionlp_completion checkpoint="outputs/2021-11-01/17-12-02/epoch\=19-step\=8719.ckpt"
```

## Evaluate Scene Graph Modification
```bash
python -m extend.eval.predict_extend input="~/.cache/extend/crowdsourced_data/test checkpoint="outputs/2021-11-01/17-12-02/epoch\=19-step\=8719.ckpt"
```

## Biomedical Event Graph Modification Dataset
All datasets can be automatically downloaded by *extend*.
Alternatively, the PC13 biomedical event graph modification dataset can be found [here](https://drive.google.com/file/d/1lHbll4xl6nFZBPACNHBnTWLDG2dAAwE2/view?usp=sharing).

## Citation
```bibtex
@inproceedings{weber-etal-2021-extend,
    title = "Extend, don't rebuild: Phrasing conditional graph modification as autoregressive sequence labelling",
    author = "Leon Weber and Samuele Garda and Jannes Münchmeyer and Ulf Leser",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    year = "2021",
    publisher = "Association for Computational Linguistics"
}
```

## License
The MIT License (MIT)
Copyright © 2021 Leon Weber

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

