"""Kucuk dogrulama: SADECE birkac anomali uzerinde retrieval kalitesini gosterir.

Amac: cekim/segmentasyon ayarlarini degistirdikten sonra tum portfoyu yeniden
islemeden, birkac temsili ornek uzerinde "prompta giden metin gercekten olayi
aciklyor mu" sorusunu hizlica gormek. Ag cagrisi yapmaz, LLM cagrisi yapmaz —
yalnizca mevcut cache + vektorindeks uzerinden calisir.

Kullanim: python probe.py
"""
from dotenv import load_dotenv
load_dotenv()

from rag.pipeline import RAGPipeline
from rag.evaluation import RETRIEVAL_QUERY, TOP_K

# (tarih, ticker, gunluk getiri) — elle secilmis temsili ornekler
PROBES = [
    ("2024-01-25", "TSLA", -12.13),   # Q4-2023 sonuclari: 2024 buyume uyarisi
    ("2021-01-27", "GME",  134.84),   # short squeeze zirvesi (haber kaynagi)
    ("2024-04-24", "TSLA",  12.06),   # Q1-2024 sonuclari
]


def main():
    p = RAGPipeline()
    for date_str, ticker, ret in PROBES:
        anomaly = {"date": date_str, "responsible_ticker": ticker,
                   "ticker_own_return_pct": ret}
        chunks = p.retrieve_for_anomaly(RETRIEVAL_QUERY, anomaly, top_k=TOP_K)
        print("=" * 78)
        print(f"{date_str}  {ticker}  {ret:+.2f}%   -> {len(chunks)} segment")
        print("=" * 78)
        for i, c in enumerate(chunks, 1):
            md = c.get("metadata", {})
            print(f"\n[{i}] {md.get('source','?')} | {str(md.get('published',''))[:10]} | "
                  f"{md.get('title','')[:70]}")
            print("    " + c.get("text", "")[:420].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
