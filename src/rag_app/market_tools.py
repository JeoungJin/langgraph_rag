# ============================================================
# 실시간 시세 조회 Tool (function calling 용)
#   - get_stock_price   : Yahoo Finance 주가
#   - get_exchange_rate : open.er-api.com 환율
# 둘 다 API 키가 필요 없습니다.
# ============================================================

from datetime import datetime

import requests
from langchain_core.tools import tool

TIMEOUT = 10


@tool
def get_stock_price(symbol: str) -> str:
    """주식의 현재가를 조회합니다. 주가 질문에만 사용하십시오.
    symbol은 Yahoo Finance 티커입니다.
    예) 삼성전자 005930.KS, SK하이닉스 000660.KS, 우리금융지주 316140.KS,
        코스닥 종목은 .KQ, 애플 AAPL, 테슬라 TSLA, 엔비디아 NVDA"""
    try:
        res = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=TIMEOUT,
        )
        if res.status_code != 200:
            return f"'{symbol}' 종목을 찾지 못했습니다. 티커를 확인해주세요."
        meta = res.json()["chart"]["result"][0]["meta"]
    except Exception as error:
        return f"주가 조회 중 오류가 발생했습니다: {error}"

    price = meta["regularMarketPrice"]
    prev = meta.get("chartPreviousClose")
    currency = meta.get("currency", "")
    when = datetime.fromtimestamp(meta["regularMarketTime"]).strftime("%Y-%m-%d %H:%M")

    text = f"{meta.get('shortName', symbol)}({symbol}) 현재가 {price:,.2f} {currency} (기준 {when})"
    if prev:
        diff = price - prev
        text += f", 전일 대비 {diff:+,.2f} ({diff / prev * 100:+.2f}%)"
    return text


@tool
def get_exchange_rate(base: str, quote: str = "KRW") -> str:
    """두 통화 사이의 환율을 조회합니다. 환율 질문에만 사용하십시오.
    base: 기준 통화 코드 (예: USD, JPY, EUR, CNY)
    quote: 상대 통화 코드 (기본 KRW)
    ※ 일본 엔(JPY) 질문이라도 1엔 기준 값이 작으므로 100엔 환산은 직접 곱해 답하십시오."""
    base, quote = base.upper(), quote.upper()
    try:
        res = requests.get(f"https://open.er-api.com/v6/latest/{base}", timeout=TIMEOUT)
        data = res.json()
        if data.get("result") != "success":
            return f"'{base}' 통화를 찾지 못했습니다."
        rate = data["rates"].get(quote)
        if rate is None:
            return f"'{quote}' 통화를 찾지 못했습니다."
    except Exception as error:
        return f"환율 조회 중 오류가 발생했습니다: {error}"

    return (
        f"1 {base} = {rate:,.4f} {quote} "
        f"(기준 {data['time_last_update_utc']}, 매매기준율 참고용이며 "
        f"은행 고시 환율과 다를 수 있음)"
    )
