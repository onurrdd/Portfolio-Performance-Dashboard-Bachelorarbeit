# Portfolio Performance Dashboard with RAG

Ein Dashboard, das die Performance eines Aktienportfolios misst, auffällige Kursbewegungen einzelner Positionen erkennt und diese Anomalien durch ein Naive-LLM sowie ein LLM mit Retrieval-Augmented Generation (RAG) erklären lässt. Die Erklärungen beider Verfahren werden mit RAGAS evaluiert und verglichen.

## Setup

```
pip install -r requirements.txt
python dashboard.py
```

Vor dem Start `.env.example` nach `.env` kopieren und die API-Schlüssel eintragen. Das Dashboard läuft anschließend unter http://127.0.0.1:8050.

## Repository Structure & Evaluation Data

- `callbacks/`: Core logic for RAG pipeline and anomaly detection.
- `data/`: Contains the evaluation results and RAGAS metrics:
  - `data/ragas_eval_20260831_031814.json`: Raw evaluation logs for the 31 analyzed portfolio anomalies (demonstrated in Chapters 6.2 and 6.3).

## Autor

Onur Darende
