# 🧠 LLM_Ontology_Population

## 🏗️ Description du projet
**LLM_Ontology_Population** est un projet de recherche visant à exploiter les **grands modèles de langage (LLMs)** pour le **peuplement automatique d’ontologies** à partir de textes non structurés.  
L’objectif est de transformer des descriptions textuelles de projets de **restauration hydro-écologique** en graphes RDF/Turtle cohérents, basés sur l’ontologie **TetraOnto**.

Approches explorées :
- **Prompt-based extraction** — extraction par prompt engineering.
- **Fine-tuning (QLoRA/LoRA)** de LLMs (ex. LLaMA 3, Qwen 3).

---

## 🎯 Objectifs
1. Automatiser la transformation de textes bruts en graphes RDF conformes à TetraOnto.  
2. Comparer la performance de différents LLMs fine-tunés.  
3. Développer une pipeline reproductible (préproc → extraction → post-traitement → évaluation).  
4. Mesurer la qualité via des évaluations quantitatives et qualitatives (manuels).

---

## 🧩 Structure du projet (racine)
LLM-Ontology-Population/
│
├── ontology/ # Schéma OWL de l’ontologie TetraOnto (TTL/OWL files)
├── outputs_results_llama3_2-3b/ # Résultats (TTL) — NE PAS pousser gros fichiers
├── outputs_results_llama3_3-70b_512/
├── outputs_results_llama3_3-70b_1024/
├── outputs_results_Qwen3_3-72b_512/
├── outputs_results_Qwen3_3-72b_1024/
├── scripts/ # Scripts: fine-tuning, génération, utilitaires
├── slurm/ # Scripts SLURM (batch) — souvent ignorés
├── test_data/ # Jeux de textes de test (NE PAS pousser si volumineux)
├── tetra_env/ # Environnement virtuel local (non versionné)
├── dataset.json # Jeu d'entraînement (peut être exclu)
├── requirements.txt # Dépendances Python
├── .gitignore # Fichier .gitignore (voir recommandations)
├── README.md # Ce fichier
└── LICENSE # Licence (MIT recommandé)


---

## ⚙️ Fichiers clés
- `scripts/` : contient les scripts de fine-tuning et d'inférence (ex. `run_llama3.py`, `generate_rdf.py`, scripts d’évaluation).  
- `ontology/` : schéma TetraOnto (OWL/Turtle).  
- `test_data/` : textes non structurés (sources).  
- `dataset.json` : dataset utilisé pour le fine-tuning (format JSONL/JSON).  
- `outputs_results_*` : dossiers contenant les `.ttl` générés par les modèles.


---

## 🔧 Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/fghazouani/LLM-Ontology-Population.git
cd LLM-Ontology-Population
