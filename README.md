# 🧬 Distributed AutoML with Genetic Algorithm on Apache Spark

An AutoML system that uses a **Genetic Algorithm** to automatically search for the best ML model and hyperparameters across a distributed computing environment built on **Apache Spark** and **Azure Databricks**.

Validated against random search baselines on two real-world datasets — Credit Card Fraud Detection and HIGGS Physics Dataset.

---

## 🚀 Overview

Manually selecting ML models and tuning hyperparameters is tedious and expertise-heavy. This system automates the entire search process using a Genetic Algorithm — a nature-inspired optimization technique that evolves a population of model configurations across generations to find high-performing solutions without exhaustive search.

Built on Spark because datasets are large and model training is expensive — Spark distributes evaluation across partitions to make the search feasible at scale.

---

## ⚙️ System Architecture

```
01_ga_core         →   Search space definition + GA evolutionary logic
02_spark_engine    →   Distributed population evaluation engine
03_pipeline_models →   Spark ML preprocessing and model construction
04_evaluation      →   Baselines, random search, scalability & parallelism tests
00_main_runner     →   End-to-end orchestration across both datasets
```

---

## 🔬 How the Genetic Algorithm Works

1. **Initialise** a population of random model configurations (chromosomes)
2. **Evaluate** each chromosome using a composite fitness function: `0.6 × AUC + 0.4 × F1`
3. **Select** top 3 configurations as parents
4. **Crossover** between parents to generate new configurations
5. **Mutate** with 30% probability to prevent premature convergence
6. **Elitism** — best configuration always carried forward unchanged
7. Repeat for N generations

Each chromosome encodes a complete model config — model type + hyperparameters:
- Logistic Regression, Random Forest, Gradient Boosted Trees, Linear SVC

---

## 📊 Datasets

| Dataset | Rows | Features | Task |
|---|---|---|---|
| Credit Card Fraud | ~284K (228K train / 56K test) | 30 | Binary classification (imbalanced, 0.17% fraud) |
| HIGGS Physics | ~550K at 5% sample (440K train / 110K test) | 28 | Binary classification |

---

## 🧪 Experiment Results

### Credit Card Fraud Detection

| Method | Best Model | Fitness | AUC | PR-AUC | Accuracy | F1 |
|---|---|---|---|---|---|---|
| Baseline (LR) | Logistic Regression | — | 0.9773 | — | — | — |
| GA | GBT | 0.9929 | 0.9886 | 0.8037 | 0.9993 | 0.9993 |
| Random Search | Random Forest | 0.9933 | 0.9893 | 0.8361 | 0.9992 | 0.9992 |

> GA and Random Search reached comparable performance (~0.993 fitness). Both significantly outperformed the untuned baseline (AUC 0.9773 → 0.9886+).

### HIGGS Physics Dataset

| Method | Best Model | Fitness |
|---|---|---|
| GA | — | 0.5781 |
| Random Search | GBT | 0.6337 |

> HIGGS is a harder classification problem. Random Search outperformed GA here — likely due to the small population size (3) causing premature convergence before the search space was adequately explored.

---

## ⚡ Scalability & Parallelism Results

### Scalability Test (HIGGS, varying data fraction)

| Data Fraction | Execution Time |
|---|---|
| 10% | 868.72 sec |
| 80% | 1037.48 sec |

> Near-linear time scaling validated — system handles increasing data size gracefully.

### Parallelism Test (Credit Card, varying partitions)

| Partitions | Execution Time |
|---|---|
| 2 | 39.41 sec |
| 16 | 99.33 sec |

> Optimal throughput at 2 partitions for this dataset size. Time increases beyond the optimal point due to Spark coordination overhead — consistent with expected distributed systems behaviour.

---

## 📦 Key Design Decisions

| Decision | Reason |
|---|---|
| Fitness = 0.6×AUC + 0.4×F1 | AUC alone is misleading on imbalanced data |
| Preprocess once per generation | Avoids recomputing Spark DAG per chromosome |
| Cache preprocessed data | Forces Spark to materialise in memory, cuts I/O |
| Random Search as baseline | Matches exact GA evaluation budget for fair comparison |
| Class weights on Credit Card | 0.17% fraud rate — handled via inverse-frequency weighting |
| PCA on HIGGS | 28 correlated physics features reduced to 10 components |
| Fixed seed=42 | Same train/test split across all chromosomes for fair comparison |
| Fault-tolerant evaluation | Per-chromosome failures caught gracefully, GA never crashes |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Distributed compute | Apache Spark, Azure Databricks |
| ML framework | Spark MLlib |
| Language | Python (PySpark) |
| Notebooks | Databricks Notebooks |

---

## 📁 Project Structure

```
distributed-automl/
├── 00_main_runner.ipynb       # Orchestration, data loading, final results
├── 01_ga_core.ipynb           # Genetic algorithm logic and search space
├── 02_spark_engine.ipynb      # Population evaluation engine
├── 03_pipeline_models.ipynb   # ML pipeline and model construction
├── 04_evaluation.ipynb        # Baselines, random search, scalability tests
└── README.md
```

---

## ⚠️ Limitations

- Small population size (3–6) risks premature convergence — evident in HIGGS results
- Single train/test split instead of cross-validation — fitness estimates may be noisy
- HIGGS uses only 5% of the full 11M row dataset
- GBT and SVM do not support class weights in Spark ML

---

## 🔗 Links

- **GitHub:** [github.com/30antara](https://github.com/30antara)
- **LinkedIn:** [linkedin.com/in/antarasinghal](https://linkedin.com/in/antarasinghal)
