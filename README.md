# Deep Learning Contextual Models for Prediction of Sport Event Outcome from Sportsman's Interviews


[![Python Version](https://img.shields.io/badge/python-3.6.3-blue.svg)](https://www.python.org)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![CC BY-SA 4.0][cc-by-sa-shield]](LICENSE)

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

This repository contains the datasets, notebooks, and supporting materials associated with the following publication:

> Boris Velichkov, Ivan Koychev, and Svetla Boytcheva.  
> *Deep learning contextual models for prediction of sport event outcome from sportsman's interviews.*  
> In *Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2019)*, 1240&ndash;1246, 2019.  
> https://doi.org/10.26615/978-954-452-056-4_142

## Citation

If you use this repository, please cite the associated paper.

```bibtex
@inproceedings{velichkov-etal-2019-deep,
    title = "Deep learning contextual models for prediction of sport event outcome from sportsman's interviews",
    author = "Velichkov, Boris and
              Koychev, Ivan and
              Boytcheva, Svetla",
    editor = "Mitkov, Ruslan and
              Angelova, Galia",
    booktitle = "Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2019)",
    month = sep,
    year = "2019",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd.",
    url = "https://aclanthology.org/R19-1142/",
    doi = "10.26615/978-954-452-056-4_142",
    pages = "1240--1246"
}
```

## Repository Structure

```
data/
├── raw/
│   ├── interviews/
│   └── structured/
├── processed/
│   ├── interviews.csv      # text dataset
│   ├── features_norm.csv   # normalized structured dataset
│   └── word2vec/
├── feature_selection/
│   ├── structured/
│   └── word2vec/
└── tokens/
    ├── words/
    ├── words_stem_1/
    ├── words_stem_2/
    └── words_stem_3/

notebooks/
├── 01_data_preparation/
├── 02_feature_selection/
├── 03_machine_learning/
├── 04_deep_learning/
├── 05_feature_analysis/
└── 06_visualization/

results/
├── figures/
├── statistics/
└── tables/
    ├── latex/
    └── source/
```

---

## Notebooks Overview

### 01\_data\_preparation

Data preparation and preprocessing notebooks:

- dataset generation from raw interviews;
- tokenization and stemming workflows;
- TF-IDF feature generation;
- normalization of structured features.

### 02\_feature\_selection

Feature-selection experiments for:

- structured features;
- TF-IDF representations with different stemming approaches.

### 03\_machine\_learning

Classical machine learning experiments:

- Naive Bayes;
- Support Vector Machines;
- repeated cross-validation evaluations.

### 04\_deep\_learning

Deep learning experiments based on BERT models.

These notebooks were originally executed in Google Colab environments.

### 05\_feature\_analysis

Additional notebooks used for:

- extended feature-selection analysis;
- dissertation tables generation;
- exporting complete feature rankings.

These notebooks are intended for analysis and reporting purposes only and do not affect the main experimental pipeline.

### 06\_visualization

Scripts for generating:

- dataset statistics;
- feature distribution plots;
- machine learning evaluation figures.

---

## Execution Order

The notebooks follow the approximate workflow below:

```
01_data_preparation
    ↓
02_feature_selection
    ↓
03_machine_learning
    ↓
04_deep_learning
```

Visualization and analysis notebooks can be executed independently after the datasets are generated.

---

## Results and Artifacts

The repository contains:

- generated figures;
- LaTeX tables;
- source TSV tables;
- statistical summaries.

Some generated LaTeX tables may contain minor manual formatting adjustments for dissertation integration and Overleaf compatibility.

---

## Requirements

The original experiments were conducted primarily using:

- Python 3.6.3

Additional analysis notebooks in `05_feature_analysis` were executed using:

- Python 3.7.4

Main Python libraries used:

- pandas
- numpy
- matplotlib
- scikit-learn
- nltk
- tensorflow
- transformers

---

## License

This repository is licensed under the CC BY-SA 4.0 License.

See the [LICENSE](LICENSE) file for details.

---

## Reproducibility Snapshot

The repository includes the tag:

```
paper-publication-snapshot
```

which corresponds to the archived state associated with the published paper.