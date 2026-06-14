from callbacks import portfolio, charts, ai_analysis, rag, naive_llm

def register_all(app, rag_pipeline):
    portfolio.register(app)
    charts.register(app)
    ai_analysis.register(app)
    rag.register(app, rag_pipeline)
    naive_llm.register(app)
