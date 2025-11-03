# 🧠 LLM-Ontology-Population

## 🏗️ Project Description
**LLM_Ontology_Population** is a research project aiming to leverage **Large Language Models (LLMs)** for the **automatic population of ontologies** from unstructured text.  
The primary goal is to convert textual descriptions of **hydro-ecological restoration operations** into structured RDF/Turtle graphs compliant with the **TetraOnto** ontology.

This project explores two approaches:
- **Prompt-based extraction** — using LLM prompts to extract ontology triples.
- **Fine-tuning (QLoRA / LoRA)** of LLMs such as *LLaMA 3* and *Qwen 3*.

---

## 🎯 Objectives
1. Automate the transformation of raw textual data into TTL graphs aligned with **TetraOnto**.  
2. Compare multiple fine-tuned LLMs on ontology population tasks.  
3. Develop a **reproducible pipeline** for training, inference, and evaluation.  
4. Evaluate the quality of extracted triples using both **automatic** and **LLM-based evaluation** methods.

---

## 🧩 Project Structure
```
LLM_Ontology_Population/
│
├── ontology/ # Contains the TetraOnto ontology schema (.owl)
│
├── test_data/ # Test datasets (raw and semi-structured texts)
│
├── outputs_results_* / # Model output folders (triplets, metrics, logs)
│
├── scripts/ # Core scripts for fine-tuning and inference
│ ├── fine_tune_llm.py # LoRA/QLoRA fine-tuning pipeline
│ ├── inference_extract_triples.py # Extraction and ontology population
│ ├── evaluate_llm_as_judge.py # Automated evaluation using LLM-as-a-Judge
│ ├── data_preprocessing.py # Dataset preprocessing and cleaning
│ ├── utils/ # Helper scripts and configuration modules
│
├── dataset.json # Fine-tuning dataset in JSON format
│
├── README.md # Project documentation (this file)
│
└── .gitignore # Files and directories excluded from Git
```


---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/fghazouani/LLM-Ontology-Population.git
cd LLM-Ontology-Population
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv tetra_env
source tetra_env/bin/activate   # (Linux/Mac)
tetra_env\Scripts\activate      # (Windows)
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Fine-tune a Model
Example using LLaMA 3 with QLoRA:
```bash
python scripts/fine_tune_llm.py --base_model llama3 --dataset dataset.json --output_dir outputs_results_llama3
```

### 5. Extract Triples and Populate the Ontology
```bash
python scripts/inference_extract_triples.py --input test_data/ --model outputs_results_llama3/
```

# 🧠 Ontology Reference

TetraOnto defines the conceptual structure of hydro-ecological restoration operations, including:

Classes: geographic zone, restoration measure, restoration operation, structure, migratory species, project owner, main contractor

Properties:
```
isManagedBy → (operation → project owner)

isLocatedOn → (structure → water body)

includes → (restoration measure → restoration operation)

associateTo → (restoration measure → structure)

hasHeight → (structure → xsd:decimal)

Each extracted text is converted into TTL triples following this schema.
```

---

# ⚙️ Technical Notes

Models can be run in FP16 or 4-bit quantization (bnb config).

Compatible GPUs: Tesla L40S, H100, and others supporting CUDA ≥ 12.1.

Designed for both research and academic reproducibility.

---

# 📫 Contact
```
Author: Fethi Ghazouani
Email: ghazouanifethi@gmail.com, fghazoua@unistra.fr
Affiliation: Université de Strasbourg – ICube Laboratory, SDC
```
⭐ If you find this project useful, please consider giving it a star on GitHub!

