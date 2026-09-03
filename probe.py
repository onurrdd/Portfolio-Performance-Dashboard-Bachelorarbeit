"""Kucuk dogrulama: SADECE birkac anomali uzerinde retrieval kalitesini gosterir.

Amac: cekim/segmentasyon ayarlarini degistirdikten sonra tum portfoyu yeniden
islemeden, birkac temsili ornek uzerinde "prompta giden metin gercekten olayi
aciklyor mu" sorusunu hizlica gormek. Ag cagrisi yapmaz, LLM cagrisi yapmaz —
yalnizca mevcut cache + vektorindeks uzerinden calisir.

Kullanim: python probe.py
"""
from dotenv import load_dotenv
load_dotenv()

from rag import config as rag_config
from rag.pipeline import RAGPipeline
from rag.evaluation import RETRIEVAL_QUERY, TOP_K

# Bearbeitete Faelle: rag/config.py (PROBE_POOL/PROBE_PICK) — EINE Quelle fuer
# Dashboard-Sparmodus, eval_probe.py und diese Datei. Auswahl aendern: dort
# PROBE_PICK anpassen. Der Tagesgewinn spielt fuer das Retrieval keine Rolle
# (nur Ticker + Zeitfenster zaehlen), wird also nicht mitgefuehrt.
PROBES = [rag_config.PROBE_POOL[i - 1] for i in rag_config.PROBE_PICK]


def main():
    p = RAGPipeline()
    for date_str, ticker in PROBES:
        anomaly = {"date": date_str, "responsible_ticker": ticker}
        chunks = p.retrieve_for_anomaly(RETRIEVAL_QUERY, anomaly, top_k=TOP_K)
        print("=" * 78)
        print(f"{date_str}  {ticker}   -> {len(chunks)} segment")
        print("=" * 78)
        for i, c in enumerate(chunks, 1):
            md = c.get("metadata", {})
            print(f"\n[{i}] {md.get('source','?')} | {str(md.get('published',''))[:10]} | "
                  f"{md.get('title','')[:70]}")
            print("    " + c.get("text", "")[:420].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
