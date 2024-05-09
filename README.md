# LLMs for step size adaptation in Evolutionary Algorithms

This repo implements the code for the paper "An investigation on the use of Large Language Models for hyperparameter tuning in Evolutionary Algorithms" by Custode et al. @ GECCO 2024.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Set your groq token in the `launch.py` file (line 7).

Run the experiments with:
```bash
python launch.py
```

## Citation

Cite this code as:
```bibtex
@inproceedings{custode2024investigation,
  title={{An investigation on the use of Large Language Models for hyperparameter tuning in Evolutionary Algorithms}},
  author={Custode, Leonardo Lucio and Yaman, Anil and Caraffini, Fabio and Iacca, Giovanni},
  booktitle={Genetic and Evolutionary Computation Conference Companion},
  publisher={ACM},
  year={2024},
  address={{Melbourne, Australia}},
}
```
