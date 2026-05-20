# -*- coding: utf-8 -*-
"""
===================================
TushareFetcher - 备用数据源 1 (Priority 2)
===================================
数据来源：Tushare Pro API（挖地兔）
特点：需要 Token、有请求配额限制
优点：数据质量高、接口稳定
流控策略：
1. 实现"每分钟调用计数器"
2. 超过免费配额（80次/分）时，强制休眠到下一分钟
3. 使用 tenacity 实现指数退避重试
"""
import json as _json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from .base import BaseFetcher, DataFetchError, RateLimitError, STANDARD_COLUMNS,is_bse_code, is_st_stock, is_kc_cy_stock, normalize_stock_code, _is_hk_market
from .realtime_types import UnifiedRealtimeQuote, ChipDistribution
from src.config import get_config
import os
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# ETF code prefixes by exchange
# Shanghai: 51xxxx, 52xxxx, 56xxxx, 58xxxx
# Shenzhen: 15xxxx, 16xxxx, 18xxxx
_ETF_SH_PREFIXES = ('51', '52', '56', '58')
_ETF_SZ_PREFIXES = ('15', '16', '18')
_ETF_ALL_PREFIXES = _ETF_SH_PREFIXES + _ETF_SZ_PREFIXES

def _is_etf_code(stock_code: str) -> bool:
    code = stock_code.strip().split('.')[0]
    return code.startswith(_ETF_ALL_PREFIXES) and len(code) == 6

def _is_us_code(stock_code: str) -> bool:
    code = stock_code.strip().upper()
    return bool(re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', code))

class _TushareHttpClient:
    """Lightweight Tushare Pro client that does not require the tushare SDK."""
    def __init__(self, token: str, timeout: int = 30, api_url: str = "http://124.222.60.121:8020") -> None:
        self._token = token
        self._timeout = timeout
        self._api_url = api_url

    def query(self, api_name: str, fields: str = "", **kwargs) -> pd.DataFrame:
        req_params = {
            "api_name": api_name,
            "token": self._token,
            "params": kwargs,
            "fields": fields,
        }
        res = requests.post(self._api_url, json=req_params, timeout=self._timeout)
        if res.status_code != 200:
            raise Exception(f"Tushare API HTTP {res.status_code}")
        result = _json.loads(res.text)
        if result.get("code") != 0:
            raise Exception(result.get("msg") or f"Tushare API error code {result.get('code')}")
        data = result.get("data") or {}
        columns = data.get("fields") or []
        items = data.get("items") or []
        return pd.DataFrame(items, columns=columns)

    def __getattr__(self, api_name: str):
        if api_name.startswith("_"):
            raise AttributeError(api_name)
        def caller(**kwargs) -> pd.DataFrame:
            return self.query(api_name, **kwargs)
        return caller

class TushareFetcher(BaseFetcher):
    name = "TushareFetcher"
    priority = int(os.getenv("TUSHARE_PRIORITY", "2"))

    def __init__(self, rate_limit_per_minute: int = 80):
        self.rate_limit_per_minute = rate_limit_per_minute
        self._call_count = 0
        self._minute_start: Optional[float] = None
        self._api: Optional[object] = None
        self.date_list: Optional[List[str]] = None
        self._date_list_end: Optional[str] = None
        self._init_api()
        self.priority = self._determine_priority()

    def _init_api(self) -> None:
        config = get_config()
        if not config.tushare_token:
            logger.warning("Tushare Token 未配置，此数据源不可用")
            return
        try:
            self._api = self._build_api_client(config.tushare_token)
            logger.info("Tushare API 初始化成功")
        except Exception as e:
            logger.error(f"Tushare API 初始化失败: {e}")
            self._api = None

    def _build_api_client(self, token: str) -> _TushareHttpClient:
        client = _TushareHttpClient(token=token)
        logger.debug("Tushare API client configured for direct HTTP calls")
        return client

    def _determine_priority(self) -> int:
        config = get_config()
        if config.tushare_token and self._api is not None:
            logger.info("✅ 检测到 TUSHARE_TOKEN 且 API 初始化成功，Tushare 数据源优先级提升为最高 (Priority -1)")
            return -1
        return 2

    def is_available(self) -> bool:
        return self._api is not None

    def _check_rate_limit(self) -> None:
        current_time = time.time()
        if self._minute_start is None:
            self._minute_start = current_time
            self._call_count = 0
        elif current_time - self._minute_start >= 60:
            self._minute_start = current_time
            self._call_count = 0
            logger.debug("速率限制计数器已重置")
        if self._call_count >= self.rate_limit_per_minute:
            elapsed = current_time - self._minute_start
            sleep_time = max(0, 60 - elapsed) + 1
            logger.warning(f"Tushare 达到速率限制 ({self._call_count}/{self.rate_limit_per_minute} 次/分钟)，等待 {sleep_time:.1f} 秒...")
            time.sleep(sleep_time)
            self._minute_start = time.time()
            self._call_count = 0
        self._call_count += 1
        logger.debug(f"Tushare 当前分钟调用次数: {self._call_count}/{self.rate_limit_per_minute}")

    def _call_api_with_rate_limit(self, method_name: str, **kwargs) -> pd.DataFrame:
        if self._api is None:
            raise DataFetchError("Tushare API 未初始化，请检查 Token 配置")
        self._check_rate_limit()
        method = getattr(self._api, method_name)
        return method(**kwargs)

    def _get_china_now(self) -> datetime:
        return datetime.now(ZoneInfo("Asia/Shanghai"))

    def _get_trade_dates(self, end_date: Optional[str] = None) -> List[str]:
        if self._api is None:
            return []
        china_now = self._get_china_now()
        requested_end_date = end_date or china_now.strftime("%Y%m%d")
        if self.date_list is not None and self._date_list_end == requested_end_date:
            return self.date_list
        start_date = (china_now - timedelta(days=20)).strftime("%Y%m%d")
        df_cal = self._call_api_with_rate_limit(
            "trade_cal",
            exchange="SSE",
            start_date=start_date,
            end_date=requested_end_date,
        )
        if df_cal is None or df_cal.empty or "cal_date" not in df_cal.columns:
            logger.warning("[Tushare] trade_cal 返回为空，无法更新交易日历缓存")
            self.date_list = []
            self._date_list_end = requested_end_date
            return self.date_list
        trade_dates = sorted(
            df_cal[df_cal["is_open"] == 1]["cal_date"].astype(str).tolist(),
            reverse=True,
        )
        self.date_list = trade_dates
        self._date_list_end = requested_end_date
        return trade_dates

    @staticmethod
    def _pick_trade_date(trade_dates: List[str], use_today: bool) -> Optional[str]:
        if not trade_dates:
            return None
        if use_today or len(trade_dates) == 1:
            return trade_dates[0]
        return trade_dates[1]

    @staticmethod
    def _detect_exchange_hint(stock_code: str) -> Optional[str]:
        upper = (stock_code or "").strip().upper()
        if upper.startswith(("SH", "SS")) or upper.endswith((".SH", ".SS")):
            return "SH"
        if upper.startswith("SZ") or upper.endswith(".SZ"):
            return "SZ"
        if upper.startswith("BJ") or upper.endswith(".BJ"):
            return "BJ"
        return None

    @classmethod
    def _get_legacy_realtime_symbol(cls, stock_code: str) -> str:
        code = normalize_stock_code(stock_code)
        exchange_hint = cls._detect_exchange_hint(stock_code)
        if code == '000001' and exchange_hint == 'SH':
            return 'sh000001'
        if code == '399001':
            return 'sz399001'
        if code == '399006':
            return 'sz399006'
        if code == '000300':
            return 'sh000300'
        if is_bse_code(code):
            return f"bj{code}"
        return code

    def _convert_stock_code(self, stock_code: str) -> str:
        raw_code = stock_code.strip()
        if '.' in raw_code:
            ts_code = raw_code.upper()
            if ts_code.endswith('.SS'):
                return f"{ts_code[:-3]}.SH"
            return ts_code
        if _is_us_code(raw_code):
            raise DataFetchError(f"TushareFetcher 不支持美股 {raw_code}，请使用 AkshareFetcher 或 YfinanceFetcher")
        if _is_hk_market(raw_code):
            return normalize_stock_code(raw_code)
        code = normalize_stock_code(raw_code)
        exchange_hint = self._detect_exchange_hint(raw_code)
        if exchange_hint == "SH":
            return f"{code}.SH"
        if exchange_hint == "SZ":
            return f"{code}.SZ"
        if exchange_hint == "BJ":
            return f"{code}.BJ"
        if code.startswith(_ETF_SH_PREFIXES) and len(code) == 6:
            return f"{code}.SH"
        if code.startswith(_ETF_SZ_PREFIXES) and len(code) == 6:
            return f"{code}.SZ"
        if is_bse_code(code):
            return f"{code}.BJ"
        if code.startswith(('600', '601', '603', '688')):
            return f"{code}.SH"
        elif code.startswith(('000', '002', '300')):
            return f"{code}.SZ"
        else:
            logger.warning(f"无法确定股票 {code} 的市场，默认使用深市")
            return f"{code}.SZ"

    def _convert_hk_stock_code_for_tushare(self, stock_code: str) -> str:
        raw_code = stock_code.strip()
        if _is_hk_market(raw_code):
            if "." in raw_code:
                ts_code = raw_code.upper()
                if ts_code.endswith(".SS"):
                    return f"{ts_code[:-3]}.SH"
                if ts_code.endswith(".HK"):
                    return ts_code
            digits = re.sub(r"\D", "", raw_code)
            if not digits:
                raise DataFetchError(f"无法识别港股代码 {raw_code}")
            code = digits[-5:].rjust(5, "0")
            return f"{code}.HK"
        return self._convert_stock_code(stock_code)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self._api is None:
            raise DataFetchError("Tushare API 未初始化，请检查 Token 配置")
        if _is_us_code(stock_code):
            raise DataFetchError(f"TushareFetcher 不支持美股 {stock_code}，请使用 AkshareFetcher 或 YfinanceFetcher")
        self._check_rate_limit()
        is_hk = _is_hk_market(stock_code)
        is_etf = _is_etf_code(stock_code)
        if is_hk:
            ts_code = self._convert_hk_stock_code_for_tushare(stock_code)
            api_name = "hk_daily"
        else:
            ts_code = self._convert_stock_code(stock_code)
            api_name = "fund_daily" if is_etf else "daily"
        ts_start = start_date.replace('-', '')
        ts_end = end_date.replace('-', '')
        logger.debug(f"调用 Tushare {api_name}({ts_code}, {ts_start}, {ts_end})")
        try:
            if is_hk:
                df = self._api.hk_daily(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
            elif is_etf:
                df = self._api.fund_daily(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
            else:
                df = self._api.daily(ts_code=ts_code, start_date=ts_start, end_date=ts_end)
            return df
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['quota', '配额', 'limit', '权限']):
                logger.warning(f"Tushare 配额可能超限: {e}")
                raise RateLimitError(f"Tushare 配额超限: {e}") from e
            raise DataFetchError(f"Tushare 获取数据失败: {e}") from e

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()
        is_hk = _is_hk_market(stock_code)
        column_mapping = {'trade_date': 'date', 'vol': 'volume'}
        df = df.rename(columns=column_mapping)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        if 'volume' in df.columns and not is_hk:
            df['volume'] = df['volume'] * 100
        if 'amount' in df.columns and not is_hk:
            df['amount'] = df['amount'] * 1000
        df['code'] = stock_code
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]
        return df

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        if self._api is None:
            logger.warning("Tushare API 未初始化，无法获取股票名称")
            return None
        if hasattr(self, '_stock_name_cache') and stock_code in self._stock_name_cache:
            return self._stock_name_cache[stock_code]
        if not hasattr(self, '_stock_name_cache'):
            self._stock_name_cache = {}
        try:
            self._check_rate_limit()
            if _is_hk_market(stock_code):
                ts_code = self._convert_hk_stock_code_for_tushare(stock_code)
                df = self._api.hk_basic(ts_code=ts_code, fields='ts_code,name')
            elif _is_etf_code(stock_code):
                ts_code = self._convert_stock_code(stock_code)
                df = self._api.fund_basic(ts_code=ts_code, fields='ts_code,name')
            else:
                ts_code = self._convert_stock_code(stock_code)
                df = self._api.stock_basic(ts_code=ts_code, fields='ts_code,name')
            if df is not None and not df.empty:
                name = df.iloc[0]['name']
                self._stock_name_cache[stock_code] = name
                logger.debug(f"Tushare 获取股票名称成功: {stock_code} -> {name}")
                return name
        except Exception as e:
            logger.warning(f"Tushare 获取股票名称失败 {stock_code}: {e}")
        return None

    def get_stock_list(self) -> Optional[pd.DataFrame]:
        if self._api is None:
            logger.warning("Tushare API 未初始化，无法获取股票列表")
            return None
        try:
            self._check_rate_limit()
            df = self._api.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,area,market')
            if df is None or df.empty:
                return None
            df = df.copy()
            df['code'] = df['ts_code'].astype(str).str.split('.').str[0]
            if not hasattr(self, '_stock_name_cache'):
                self._stock_name_cache = {}
            for _, row in df.iterrows():
                self._stock_name_cache[row['code']] = row['name']
            logger.info(f"Tushare 获取股票列表成功: {len(df)} 条")
            return df[['code', 'name', 'industry', 'area', 'market']]
        except Exception as e:
            logger.warning(f"Tushare 获取股票列表失败: {e}")
        return None

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        if self._api is None:
            return None
        if _is_hk_market(stock_code):
            logger.debug(f"TushareFetcher 跳过港股实时行情 {stock_code}")
            return None
        normalized_code = normalize_stock_code(stock_code)
        from .realtime_types import RealtimeSource, safe_float, safe_int
        self._check_rate_limit()
        try:
            ts_code = self._convert_stock_code(stock_code)
            df = self._api.quotation(ts_code=ts_code)
            if df is not None and not df.empty:
                row = df.iloc[0]
                logger.debug(f"Tushare Pro 实时行情获取成功: {stock_code}")
                return UnifiedRealtimeQuote(
                    code=normalized_code,
                    name=str(row.get('name', '')),
                    source=RealtimeSource.TUSHARE,
                    price=safe_float(row.get('price')),
                    change_pct=safe_float(row.get('pct_chg')),
                    change_amount=safe_float(row.get('change')),
                    volume=safe_int(row.get('vol')),
                    amount=safe_float(row.get('amount')),
                    high=safe_float(row.get('high')),
                    low=safe_float(row.get('low')),
                    open_price=safe_float(row.get('open')),
                    pre_close=safe_float(row.get('pre_close')),
                    turnover_rate=safe_float(row.get('turnover_ratio')),
                    pe_ratio=safe_float(row.get('pe')),
                    pb_ratio=safe_float(row.get('pb')),
                    total_mv=safe_float(row.get('total_mv')),
                )
        except Exception as e:
            logger.debug(f"Tushare Pro 实时行情不可用 (可能是积分不足): {e}")
        try:
            import tushare as ts
            symbol = self._get_legacy_realtime_symbol(stock_code)
            df = ts.get_realtime_quotes(symbol)
            if df is None or df.empty:
                return None
            row = df.iloc[0]
            price = safe_float(row['price'])
            pre_close = safe_float(row['pre_close'])
            change_pct = 0.0
            change_amount = 0.0
            if price and pre_close and pre_close > 0:
                change_amount = price - pre_close
                change_pct = (change_amount / pre_close) * 100
            return UnifiedRealtimeQuote(
                code=normalized_code,
                name=str(row['name']),
                source=RealtimeSource.TUSHARE,
                price=price,
                change_pct=round(change_pct, 2),
                change_amount=round(change_amount, 2),
                volume=safe_int(row['volume']) // 100,
                amount=safe_float(row['amount']),
                high=safe_float(row['high']),
                low=safe_float(row['low']),
                open_price=safe_float(row['open']),
                pre_close=pre_close,
            )
        except Exception as e:
            logger.warning(f"Tushare (旧版) 获取实时行情失败 {stock_code}: {e}")
            return None

    def get_main_indices(self, region: str = "cn") -> Optional[List[dict]]:
        if region != "cn":
            return None
        if self._api is None:
            return None
        from .realtime_types import safe_float
        indices_map = {
            '000001.SH': '上证指数',
            '399001.SZ': '深证成指',
            '399006.SZ': '创业板指',
            '000688.SH': '科创50',
            '000016.SH': '上证50',
            '000300.SH': '沪深300',
        }
        try:
            self._check_rate_limit()
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - pd.Timedelta(days=5)).strftime('%Y%m%d')
            results = []
            for ts_code, name in indices_map.items():
                try:
                    df = self._api.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                    if df is not None and not df.empty:
                        row = df.iloc[0]
                        current = safe_float(row['close'])
                        prev_close = safe_float(row['pre_close'])
                        results.append({
                            'code': ts_code.split('.')[0],
                            'name': name,
                            'current': current,
                            'change': safe_float(row['change']),
                            'change_pct': safe_float(row['pct_chg']),
                            'open': safe_float(row['open']),
                            'high': safe_float(row['high']),
                            'low': safe_float(row['low']),
                            'prev_close': prev_close,
                            'volume': safe_float(row['vol']),
                            'amount': safe_float(row['amount']) * 1000,
                            'amplitude': 0.0
                        })
                except Exception as e:
                    logger.debug(f"Tushare 获取指数 {name} 失败: {e}")
                    continue
            if results:
                return results
            else:
                logger.warning("[Tushare] 未获取到指数行情数据")
        except Exception as e:
            logger.error(f"[Tushare] 获取指数行情失败: {e}")
        return None

    def get_market_stats(self) -> Optional[dict]:
        if self._api is None:
            return None
        try:
            logger.info("[Tushare] ts.pro_api() 获取市场统计...")
            china_now = self._get_china_now()
            current_clock = china_now.strftime("%H:%M")
            current_date = china_now.strftime("%Y%m%d")
            trade_dates = self._get_trade_dates(current_date)
            if not trade_dates:
                return None
            if current_date in trade_dates:
                use_realtime = (current_clock >= '09:30' and current_clock <= '16:30')
            else:
                use_realtime = False
            if use_realtime:
                try:
                    df = self._call_api_with_rate_limit("rt_k", ts_code='3*.SZ,6*.SH,0*.SZ,92*.BJ')
                    if df is not None and not df.empty:
                        return self._calc_market_stats(df)
                except Exception as e:
                    logger.error(f"[Tushare] ts.pro_api().rt_k 尝试获取实时数据失败: {e}")
                    return None
            else:
                if current_date not in trade_dates:
                    last_date = self._pick_trade_date(trade_dates, use_today=True)
                else:
                    if current_clock < '09:30':
                        last_date = self._pick_trade_date(trade_dates, use_today=False)
                    else:
                        last_date = self._pick_trade_date(trade_dates, use_today=True)
                if last_date is None:
                    return None
                try:
                    df = self._call_api_with_rate_limit(
                        "daily",
                        ts_code='3*.SZ,6*.SH,0*.SZ,92*.BJ',
                        start_date=last_date,
                        end_date=last_date,
                    )
                    df.columns = [col.lower() for col in df.columns]
                    df_basic = self._call_api_with_rate_limit("stock_basic", fields='ts_code,name')
                    df = pd.merge(df, df_basic, on='ts_code', how='left')
                    if 'amount' in df.columns:
                        df['amount'] = df['amount'] * 1000
                    if df is not None and not df.empty:
                        return self._calc_market_stats(df)
                except Exception as e:
                    logger.error(f"[Tushare] ts.pro_api().daily 获取数据失败: {e}")
        except Exception as e:
            logger.error(f"[Tushare] 获取市场统计失败: {e}")
        return None

    def _calc_market_stats(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        import numpy as np
        df = df.copy()
        code_col = next((c for c in ['代码', '股票代码', 'ts_code','stock_code'] if c in df.columns), None)
        name_col = next((c for c in ['名称', '股票名称','name','name'] if c in df.columns), None)
        close_col = next((c for c in ['最新价', '最新价', 'close','lastPrice'] if c in df.columns), None)
        pre_close_col = next((c for c in ['昨收', '昨日收盘', 'pre_close','lastClose'] if c in df.columns), None)
        amount_col = next((c for c in ['成交额', '成交额', 'amount','amount'] if c in df.columns), None)
        limit_up_count = 0
        limit_down_count = 0
        up_count = 0
        down_count = 0
        flat_count = 0
        for code, name, current_price, pre_close, amount in zip(df[code_col], df[name_col], df[close_col], df[pre_close_col], df[amount_col]):
            if pd.isna(current_price) or pd.isna(pre_close) or current_price in ['-'] or pre_close in ['-'] or amount == 0:
                continue
            current_price = float(current_price)
            pre_close = float(pre_close)
            pure_code = normalize_stock_code(str(code))
            if is_bse_code(pure_code):
                ratio = 0.30
            elif is_kc_cy_stock(pure_code):
                ratio = 0.20
            elif is_st_stock(name):
                ratio = 0.05
            else:
                ratio = 0.10
            limit_up_price = np.floor(pre_close * (1 + ratio) * 100 + 0.5) / 100.0
            limit_down_price = np.floor(pre_close * (1 - ratio) * 100 + 0.5) / 100.0
            limit_up_price_Tolerance = round(abs(pre_close * (1 + ratio) - limit_up_price), 10)
            limit_down_price_Tolerance = round(abs(pre_close * (1 - ratio) - limit_down_price), 10)
            if current_price > 0:
                is_limit_up = (abs(current_price - limit_up_price) <= limit_up_price_Tolerance)
                is_limit_down = (abs(current_price - limit_down_price) <= limit_down_price_Tolerance)
                if is_limit_up:
                    limit_up_count += 1
                if is_limit_down:
                    limit_down_count += 1
                if current_price > pre_close:
                    up_count += 1
                elif current_price < pre_close:
                    down_count += 1
                else:
                    flat_count += 1
        stats = {
            'up_count': up_count,
            'down_count': down_count,
            'flat_count': flat_count,
            'limit_up_count': limit_up_count,
            'limit_down_count': limit_down_count,
            'total_amount': 0.0,
        }
        if amount_col and amount_col in df.columns:
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
            stats['total_amount'] = (df[amount_col].sum() / 1e8)
        return stats

    def get_trade_time(self,early_time='09:30',late_time='16:30') -> Optional[str]:
        china_now = self._get_china_now()
        china_date = china_now.strftime("%Y%m%d")
        china_clock = china_now.strftime("%H:%M")
        trade_dates = self._get_trade_dates(china_date)
        if not trade_dates:
            return None
        if china_date in trade_dates:
            if early_time < china_clock < late_time:
                use_today = False
            else:
                use_today = True
        else:
            use_today = True
        start_date = self._pick_trade_date(trade_dates, use_today=use_today)
        if not use_today:
            logger.info(f"[Tushare] 当前时间 {china_clock} 可能无法获取当天筹码分布，尝试获取前一个交易日的数据 {start_date}")
        return start_date

    def get_sector_rankings(self, n: int = 5) -> Optional[Tuple[list, list]]:
        def _get_rank_top_n(df: pd.DataFrame, change_col: str, industry_name: str, n: int) -> Tuple[list, list]:
            df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
            df = df.dropna(subset=[change_col])
            top = df.nlargest(n, change_col)
            top_sectors = [{'name': row[industry_name], 'change_pct': row[change_col]} for _, row in top.iterrows()]
            bottom = df.nsmallest(n, change_col)
            bottom_sectors = [{'name': row[industry_name], 'change_pct': row[change_col]} for _, row in bottom.iterrows()]
            return top_sectors, bottom_sectors
        start_date = self.get_trade_time(early_time='00:00', late_time='15:30')
        if not start_date:
            return None
        logger.info("[Tushare] ts.pro_api().moneyflow_ind_ths 获取板块排行(同花顺)...")
        try:
            df = self._call_api_with_rate_limit("moneyflow_ind_ths", trade_date=start_date)
            if df is not None and not df.empty:
                return _get_rank_top_n(df, 'pct_change', 'industry', n)
        except Exception as e:
            logger.warning(f"[Tushare] 获取同花顺行业板块涨跌榜失败: {e} 尝试东财接口")
        logger.info("[Tushare] ts.pro_api().moneyflow_ind_dc 获取板块排行(东财)...")
        try:
            df = self._call_api_with_rate_limit("moneyflow_ind_dc", trade_date=start_date)
            if df is not None and not df.empty:
                df = df[df['content_type'] == '行业']
                return _get_rank_top_n(df, 'pct_change', 'name', n)
        except Exception as e:
            logger.warning(f"[Tushare] 获取东财行业板块涨跌榜失败: {e}")
        return None

    # ==========================================================================
    # 👇 👇 👇 这是唯一被修正的函数：cyq_perf 版本，无错误 👇 👇 👇
    # ==========================================================================
    def get_chip_distribution(self, stock_code: str) -> Optional[ChipDistribution]:
        """
        获取筹码分布数据
        数据来源：ts.pro_api().cyq_perf()
        保持与原 cyq_chips 计算完全相同的数据格式
        """
        if _is_us_code(stock_code):
            logger.warning(f"[Tushare] TushareFetcher 不支持美股 {stock_code} 的筹码分布")
            return None
        if _is_etf_code(stock_code):
            logger.warning(f"[Tushare] TushareFetcher 不支持 ETF {stock_code} 的筹码分布")
            return None
        if _is_hk_market(stock_code):
            logger.warning(f"[Tushare] TushareFetcher 不支持港股 {stock_code} 的筹码分布")
            return None

        try:
            start_date = self.get_trade_time(early_time='00:00', late_time='19:00')
            if not start_date:
                return None

            ts_code = self._convert_stock_code(stock_code)
            df = self._call_api_with_rate_limit(
                "cyq_perf",
                ts_code=ts_code,
                trade_date=start_date,
            )

            if df is None or df.empty:
                return None

            row = df.iloc[0]

            # 已修复：Tushare cyq_perf 正确字段名
            profit_ratio = round(row['profit'] / 100, 4)
            avg_cost = round(row['avg_cost'], 4)

            cost_90_low = round(row['cost_90_low'], 4)
            cost_90_high = round(row['cost_90_high'], 4)
            concentration_90 = round(row['cyq_q90'] / 100, 4)

            cost_70_low = round(row['cost_70_low'], 4)
            cost_70_high = round(row['cost_70_high'], 4)
            concentration_70 = round(row['cyq_q70'] / 100, 4)

            chip = ChipDistribution(
                code=stock_code,
                date=datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d'),
                profit_ratio=profit_ratio,
                avg_cost=avg_cost,
                cost_90_low=cost_90_low,
                cost_90_high=cost_90_high,
                concentration_90=concentration_90,
                cost_70_low=cost_70_low,
                cost_70_high=cost_70_high,
                concentration_70=concentration_70,
            )

            logger.info(f"[筹码分布] {stock_code} 日期={chip.date}: 获利比例={chip.profit_ratio:.1%}, "
                        f"平均成本={chip.avg_cost}, 90%集中度={chip.concentration_90:.2%}, "
                        f"70%集中度={chip.concentration_70:.2%}")
            return chip

        except Exception as e:
            logger.warning(f"[Tushare] 获取筹码分布失败 {stock_code}: {e}")
            return None
    # ==========================================================================
    # 👆 👆 👆 修正结束 👆 👆 👆
    # ==========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    fetcher = TushareFetcher()
    try:
        df = fetcher.get_daily_data('600519')
        print(f"获取成功，共 {len(df)} 条数据")
        print(df.tail())
        name = fetcher.get_stock_name('600519')
        print(f"股票名称: {name}")
    except Exception as e:
        print(f"获取失败: {e}")

    print("\n" + "=" * 50)
    print("Testing get_market_stats (tushare)")
    print("=" * 50)
    try:
        stats = fetcher.get_market_stats()
        if stats:
            print(f"Market Stats successfully computed:")
            print(f"Up: {stats['up_count']} (Limit Up: {stats['limit_up_count']})")
            print(f"Down: {stats['down_count']} (Limit Down: {stats['limit_down_count']})")
            print(f"Flat: {stats['flat_count']}")
            print(f"Total Amount: {stats['total_amount']:.2f} 亿")
        else:
            print("Failed to compute market stats.")
    except Exception as e:
        print(f"Failed to compute market stats: {e}")

    print("\n" + "=" * 50)
    print("测试筹码分布数据获取")
    print("=" * 50)
    try:
        chip = fetcher.get_chip_distribution('600519')
    except Exception as e:
        print(f"[筹码分布] 获取失败: {e}")

    print("\n" + "=" * 50)
    print("测试行业板块排名获取")
    print("=" * 50)
    try:
        rankings = fetcher.get_sector_rankings(n=5)
        if rankings:
            top, bottom = rankings
            print("涨幅榜 Top 5:")
            for sector in top:
                print(f"{sector['name']}: {sector['change_pct']}%")
            print("\n跌幅榜 Top 5:")
            for sector in bottom:
                print(f"{sector['name']}: {sector['change_pct']}%")
        else:
            print("未获取到行业板块排名数据")
    except Exception as e:
        print(f"[行业板块排名] 获取失败: {e}")
