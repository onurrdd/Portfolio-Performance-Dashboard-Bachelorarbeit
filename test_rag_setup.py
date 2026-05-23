#!/usr/bin/env python
"""Test RAG pipeline setup and initialization"""

import sys
import os

print("=" * 60)
print("RAG Setup Test")
print("=" * 60)

# Test 1: FAISS import
print("\n[1/5] Testing FAISS import...")
try:
    import faiss
    print("✅ FAISS imported successfully")
except ImportError as e:
    print(f"❌ FAISS import failed: {e}")
    sys.exit(1)

# Test 2: Sentence-transformers import
print("\n[2/5] Testing sentence-transformers import...")
try:
    from sentence_transformers import SentenceTransformer
    print("✅ Sentence-transformers imported successfully")
except ImportError as e:
    print(f"❌ Sentence-transformers import failed: {e}")
    sys.exit(1)

# Test 3: RAG modules import
print("\n[3/5] Testing RAG modules import...")
try:
    from rag.vectorstore import FAISSVectorStore
    from rag.pipeline import RAGPipeline
    print("✅ RAG modules imported successfully")
except ImportError as e:
    print(f"❌ RAG modules import failed: {e}")
    sys.exit(1)

# Test 4: FAISS Vectorstore initialization
print("\n[4/5] Testing FAISS vectorstore initialization...")
try:
    os.makedirs("./data/faiss", exist_ok=True)
    vectorstore = FAISSVectorStore(persist_dir="./data/faiss", embedding_dim=384)
    print(f"✅ FAISS vectorstore initialized. Current size: {vectorstore.count_documents()}")
except Exception as e:
    print(f"❌ FAISS vectorstore initialization failed: {e}")
    sys.exit(1)

# Test 5: RAG Pipeline initialization
print("\n[5/5] Testing RAG pipeline initialization...")
try:
    pipeline = RAGPipeline(persist_dir="./data/faiss")
    stats = pipeline.get_stats()
    print(f"✅ RAG Pipeline initialized successfully")
    print(f"   - Indexed documents: {stats.get('indexed_documents', 0)}")
    print(f"   - Embedding model: {stats.get('embedding_model', 'N/A')}")
    print(f"   - Embedding dimension: {stats.get('embedding_dimension', 'N/A')}")
except Exception as e:
    print(f"❌ RAG Pipeline initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed! RAG setup is complete.")
print("=" * 60)
