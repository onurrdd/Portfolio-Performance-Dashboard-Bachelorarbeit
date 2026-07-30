import json

# Sprache der LLM-ANTWORT: "de", "en" oder "tr".
# Die Prompts selbst sind IMMER Englisch (Projektregel, siehe CLAUDE.md).
RESPONSE_LANGUAGE = "tr"

# "en" fehlt bewusst: der Prompt ist bereits Englisch, eine Extra-Anweisung wäre redundant.
RESPONSE_LANGUAGE_INSTRUCTIONS = {
    "de": "\n\nPlease respond in German.",
    "tr": "\n\nPlease respond in Turkish.",
}

ADVISOR_PROMPT_TEMPLATE = """You are a portfolio advisor. Answer the questions based on the information provided.

Evaluate the current portfolio's performance in light of the information given. Provide clear, concrete answers.

Information: Return and Sharpe ratio of the portfolio and of the S&P 500 index

{performance_info}

What are the financial market reasons for this success or underperformance?
Which tickers is the result driven by? What are the reasons for the performance deviations in those tickers?
Cite sources for the reasons you present.

Information: Points where the portfolio deviated from the S&P 500, the tickers responsible for those deviations, and their prices before and after the deviation

{deviation_info}"""

_MISSING_INFO_PLACEHOLDER = "(not provided yet)"


def _format_info(info):
    if info is None:
        return _MISSING_INFO_PLACEHOLDER
    if isinstance(info, (dict, list)):
        return json.dumps(info, ensure_ascii=False, indent=2)
    return str(info)


def performance_info_from_analysis_data(analysis_data):
    if not analysis_data:
        return None

    metrics = analysis_data.get('metrics', {})
    rolling_sharpe = analysis_data.get('rolling_sharpe', {})
    benchmark = analysis_data.get('benchmark', {})

    return {
        'portfolio': {
            'total_return_pct': metrics.get('total_return'),
            'annualized_return_pct': metrics.get('annualized_return'),
            'sharpe_ratio': rolling_sharpe.get('current'),
        },
        'sp500': {
            'total_return_pct': benchmark.get('total_return'),
            'sharpe_ratio': benchmark.get('sharpe_current'),
        },
    }


def build_advisor_prompt(performance_info=None, deviation_info=None, news_context=None, user_question=None):
    prompt = ADVISOR_PROMPT_TEMPLATE.format(
        performance_info=_format_info(performance_info),
        deviation_info=_format_info(deviation_info),
    )

    if news_context is not None:
        prompt += f"\n\nAdditional context (news):\n{_format_info(news_context)}"

    if user_question is not None:
        prompt += f"\n\nUser question: {user_question}"

    if RESPONSE_LANGUAGE in RESPONSE_LANGUAGE_INSTRUCTIONS:
        prompt += RESPONSE_LANGUAGE_INSTRUCTIONS[RESPONSE_LANGUAGE]

    return prompt
