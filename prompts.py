import json

ADVISOR_PROMPT_TEMPLATE = """Sen bir portföy danışmanısın. Verilen bilgiler ışığında soruları yanıtla.

Mevcut portföyün başarısını verilen bilgiler ışığında değerlendir. Net yanıtlar ver.

Bilgi: Portföyün ve SP500 Index'inin

getiri oranı
Sharpe Ratiosu

{performance_info}

Başarının veya başarısızlığın
Finansal piyasa sebepleri nelerdir?
Sonuç hangi tickerardan kaynaklanmaktadır? Bu tickerlardaki performans sapmalarının sebepleri nelerdir?
Sebep olarak sunduğun bilgiler için kaynak ver.

Bilgi: Portföyün Sp500'den saptığı noktalar. O noktaların sebebi olan ticker'lar ve sapma noktasından önceki ve sonraki fiyatları

{deviation_info}"""

_MISSING_INFO_PLACEHOLDER = "(henüz sağlanmadı)"


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

    return {
        'portfolio': {
            'total_return_pct': metrics.get('total_return'),
            'annualized_return_pct': metrics.get('annualized_return'),
            'sharpe_ratio': rolling_sharpe.get('current'),
        },
        'sp500': {
            'total_return_pct': None,
            'sharpe_ratio': None,
        },
    }


def build_advisor_prompt(performance_info=None, deviation_info=None, news_context=None, user_question=None):
    prompt = ADVISOR_PROMPT_TEMPLATE.format(
        performance_info=_format_info(performance_info),
        deviation_info=_format_info(deviation_info),
    )

    if news_context is not None:
        prompt += f"\n\nZusätzlicher Kontext (Nachrichten):\n{_format_info(news_context)}"

    if user_question is not None:
        prompt += f"\n\nBenutzerfrage: {user_question}"

    return prompt
