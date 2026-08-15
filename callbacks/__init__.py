from callbacks import portfolio, charts, ai_analysis, rag, naive_llm

def register_all(app, rag_provider, initial_positions, run_id):
    portfolio.register(app, initial_positions, run_id)
    charts.register(app)
    ai_analysis.register(app)
    rag.register(app, rag_provider)
    naive_llm.register(app)
