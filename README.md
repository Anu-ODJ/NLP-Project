# NLP for Museum Metadata: Artwork Classification and Discovery

**Project focus:** Natural Language Processing (NLP) · Text classification · Cultural heritage data · Imbalanced learning

This project investigates how textual metadata from museum collections can be used to classify artworks and improve the discoverability of digitised cultural heritage. Using data from the **Harvard Art Museums API**, it compares traditional machine-learning approaches with neural and transformer-based models.

The central question is practical: can titles, descriptions, techniques and other curator-supplied text help classify artworks when museum metadata is incomplete, inconsistent and unevenly distributed across categories?

> **Research scope:** This is a text-based classification project. It uses museum metadata rather than artwork image pixels as the model input.

## Why this problem matters

Museum collections are increasingly available through digital catalogues and APIs, but their metadata can be incomplete, inconsistent or non-standardised. This makes collection search and cross-collection analysis harder for researchers, curators, educators and the public.

NLP-based classification could support:
- suggesting or filling missing collection categories for curator review;
- improving search and discovery across digitised collections;
- reducing repetitive manual annotation work;
- enabling more consistent metadata across institutions; and
- supporting research into representation and the discoverability of underrepresented artists and works.

Automated predictions should be treated as decision support, not as authoritative art-historical judgements. Museum taxonomies and descriptions can reflect historical and institutional biases.

## Dataset

The project uses the **Harvard Art Museums API**. The presentation describes the API collection as containing approximately 250,000 artwork records, not all of which have complete metadata.

Text fields considered for classification include:
- `title`
- `description`
- `technique`

The target field is the museum's artwork `classification`, with examples including Paintings, Prints and Sculpture.

The presentation identifies several data challenges:
- missing or incomplete text fields;
- inconsistent descriptive practices;
- short and noisy text;
- class imbalance; and
- differences in how collections are catalogued.

**Important dataset qualification:** the presentation does not establish that all approximately 250,000 API records were used in model training. The exploratory class-distribution chart shows a much smaller analysed sample. The exact extraction date, filtering rules, sample size and train/validation/test split should be documented alongside the notebooks before the experiment is treated as fully reproducible.

Dataset source: [Harvard Art Museums API documentation](https://github.com/harvardartmuseums/api-docs).

## Approach

### 1. Data preparation and exploration

The reported preprocessing workflow includes:
- lowercasing and tokenisation;
- punctuation and special-character handling;
- stop-word removal;
- rare-word handling and lemmatisation;
- TF-IDF feature extraction for traditional models;
- tokenisation and sequence padding for neural models; and
- consideration of class weighting and over/undersampling to address imbalance.

The presentation reports a vocabulary size of 8,914 in one exploratory analysis. It later reports a vocabulary size of 27,434 for the LSTM/GRU pipeline, indicating that model-specific preprocessing or data configurations should be made explicit in the reproducibility materials.

### 2. Models compared

| Model | Role in the comparison |
|---|---|
| Multinomial Naïve Bayes | Efficient traditional text-classification baseline using sparse text features |
| Linear Support Vector Machine (SVM) | Strong traditional classifier; class weighting was used to improve minority-class recall |
| LSTM/GRU | Sequential neural approach for learning patterns in token sequences |
| DistilBERT | Fine-tuned pretrained transformer using contextual representations and attention |

The presentation also discusses other candidate approaches, including Logistic Regression and Random Forest, but the main reported comparison is between the four models above.

### 3. Evaluation

The project reports using:
- accuracy;
- weighted F1-score;
- per-class precision, recall and F1;
- confusion matrices;
- ROC curves / ROC-AUC; and
- precision-recall curves / PR-AUC.

This multi-metric approach matters because accuracy alone can conceal poor performance on minority classes.

## Results reported in the presentation

The summary table on page 17 gives the following performance figures:

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Multinomial Naïve Bayes | 95.6% | 0.9491 |
| Linear SVM | 98.3% | 0.9827 |
| LSTM/GRU | 87.4% | 0.8152 |
| DistilBERT | 98.6% | 0.9859 |

These values are transcribed from the presentation's summary table and should be treated as **reported project results**, not independently reproduced benchmarks.

### What the results suggest

- **DistilBERT** is identified in the presentation's discussion and conclusion as the strongest overall model, particularly for minority-class performance.
- **Linear SVM with `class_weight='balanced'`** is a strong traditional alternative. The presentation reports improved minority-class recall despite a small reduction in overall accuracy after class weighting.
- **Multinomial Naïve Bayes** provides a computationally efficient baseline, with tuning of `alpha=0.1` improving its reported accuracy and weighted F1.
- **LSTM/GRU** struggled more with minority classes in the reported experiment, suggesting that its configuration and tuning were not sufficient for this dataset.

The confusion matrices and precision-recall curves in the presentation (pages 18–21) are particularly relevant: they show why class-level performance needs to be examined rather than relying only on aggregate accuracy.

### Result discrepancies to resolve

The presentation contains more than one set of reported results:
- Page 16 gives DistilBERT as 99.5% accuracy and 0.9946 weighted F1, whereas the page 17 summary table gives 98.6% and 0.9859.
- Page 16 gives LSTM/GRU as 92.5% accuracy and 0.9247 weighted F1, whereas the page 17 summary table gives 87.4% and 0.8152.
- The objectives refer to an expected majority-class baseline of approximately 46%, while the exploratory analysis on page 9 reports 87.38% for always predicting Prints.
- The presentation uses different class examples in different sections (including Architecture in the deep-learning discussion, and Paintings / Prints / Sculpture elsewhere).

These figures may correspond to different runs, subsets, label mappings or evaluation splits. The presentation does not provide enough detail to determine which explanation is correct. The notebook outputs and experiment configuration should be checked before selecting a definitive headline score.

## Key interpretation

The project illustrates a useful trade-off in applied NLP:

- A pretrained transformer can capture contextual information that simpler bag-of-words methods may miss.
- A linear SVM can remain highly competitive while offering a simpler and more interpretable feature space.
- Class imbalance changes the meaning of an apparently strong accuracy score.
- Model comparisons are only meaningful when the same data split, target-label mapping and evaluation procedure are used.

## Responsible and ethical considerations

Museum metadata is not neutral ground truth in every interpretive sense. Descriptions, naming conventions, provenance records and classification systems are produced within particular historical and institutional contexts.

Potential risks include:
- reproducing omissions or biases already present in collection catalogues;
- systematically misclassifying works from underrepresented traditions;
- treating model predictions as definitive curatorial decisions; and
- making cross-institution comparisons that ignore differences in taxonomies and cataloguing practice.

A responsible deployment would retain human review, document uncertainty, monitor per-class performance and involve relevant curatorial and community expertise when changing collection metadata.

## Reproducibility checklist

The presentation describes the research design and reports results, but the following details should accompany the code before the experiment is considered independently reproducible:

- [ ] API extraction date, query parameters and raw-data provenance
- [ ] Exact number of records retrieved, filtered and retained
- [ ] Missing-value handling and text-field concatenation rules
- [ ] Final target labels and label mapping
- [ ] Train/validation/test split, stratification and random seed
- [ ] Confirmation that all models use the same evaluation split
- [ ] Full preprocessing and feature-extraction configuration per model
- [ ] Hyperparameter search spaces and selected configurations
- [ ] Per-class metrics and saved confusion matrices
- [ ] Resolution of the result discrepancies noted above
- [ ] Dependency versions and instructions for rerunning the notebooks

## Potential next steps

1. Re-run all models against one documented, stratified split and reconcile the conflicting metrics.
2. Report macro F1 and per-class recall alongside weighted F1 to make minority-class performance more visible.
3. Add calibration or confidence thresholds so uncertain predictions can be routed to curators.
4. Evaluate on a separate collection or institution to test whether the approach generalises across cataloguing systems.
5. Explore multimodal or knowledge-graph approaches as extensions, while keeping the current work clearly scoped as metadata-based NLP.

## Repository contents

The repository should ultimately contain:
- notebooks for data acquisition, exploration, preprocessing and each model;
- a requirements file or environment specification;
- configuration and evaluation scripts;
- compact result tables and visualisations; and
- instructions for obtaining the data from the Harvard Art Museums API.

Large raw datasets and model checkpoints should not be committed unless licensing, size and repository constraints have been checked. Prefer documented data-acquisition steps and lightweight, reproducible outputs.

## References

- Harvard Art Museums. (2025). [API documentation](https://github.com/harvardartmuseums/api-docs).
- Sporleder, C. (2010). Natural language processing for cultural heritage domains. *Language and Linguistics Compass, 4*(9), 750–768.
- Castellano, G. & Vessio, G. (2021). Deep learning approaches to pattern extraction and recognition in paintings and drawings: An overview. *Neural Computing and Applications, 33*(19), 12263–12282.
- Conde, M. V. & Turgutlu, K. (2021). CLIP-Art: Contrastive pre-training for fine-grained art classification.
- Brahim El Vaigh, C. et al. (2021). GCNBoost: Artwork classification by label propagation through a knowledge graph.
- Frank, E. & Bouckaert, R. R. (2006). Naïve Bayes for text classification with unbalanced classes.

---

**Project author:** Anuoluwapo Oloja  
**Institution:** University of Hull
