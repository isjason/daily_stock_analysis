# -*- coding: utf-8 -*-
"""
===================================
AkshareFetcher - 主数据源 (Priority 1)
===================================

数据来源：东方财富爬虫（通过 akshare 库）
集成策略：使用 efinance 动态修复 A 股股票名称缺失问题
"""

import logging
import random
import time
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import efinance as ef  # 新增依赖：pip install efinance
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .base import BaseFetcher, DataFetchError, RateLimitError, STANDARD_COLUMNS


@dataclass
class RealtimeQuote:
    """实时行情数据容器"""
    code: str
    name: str = ""
    price: float = 0.0
    change_pct: float = 0.0
    change_amount: float = 0.0
    volume_ratio: float = 0.0
    turnover_rate: float = 0.0
    amplitude: float = 0.0
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    total_mv: float = 0.0
    circ_mv: float = 0.0
    change_60d: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'price': self.price,
            'change_pct': self.change_pct,
            'volume_ratio': self.volume_ratio,
            'turnover_rate': self.turnover_rate,
            'amplitude': self.amplitude,
            'pe_ratio': self.pe_ratio,
            'pb_ratio': self.pb_ratio,
            'total_mv': self.total_mv,
            'circ_mv': self.circ_mv,
            'change_60d': self.change_60d,
        }


@dataclass  
class ChipDistribution:
    """筹码分布数据容器"""
    code: str
    date: str = ""
    profit_ratio: float = 0.0
    avg_cost: float = 0.0
    cost_90_low: float = 0.0
    cost_90_high: float = 0.0
    concentration_90: float = 0.0
    cost_70_low: float = 0.0
    cost_70_high: float = 0.0
    concentration_70: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'date': self.date,
            'profit_ratio': self.profit_ratio,
            'avg_cost': self.avg_cost,
            'cost_90_low': self.cost_90_low,
            'cost_90_high': self.cost_90_high,
            'concentration_90': self.concentration_90,
            'concentration_70': self.concentration_70,
        }

logger = logging.getLogger(__name__)

# 全局缓存与配置
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

_realtime_cache = {'data': None, 'timestamp': 0, 'ttl': 60}
_etf_realtime_cache = {'data': None, 'timestamp': 0, 'ttl': 60}


def _is_etf_code(stock_code: str) -> bool:
    etf_prefixes = ('51', '52', '56', '58', '15', '16', '18')
    return stock_code.startswith(etf_prefixes) and len(stock_code) == 6


def _is_hk_code(stock_code: str) -> bool:
    code = stock_code.lower()
    if code.startswith('hk'):
        numeric_part = code[2:]
        return numeric_part.isdigit() and 1 <= len(numeric_part) <= 5
    return code.isdigit() and len(code) == 5


class AkshareFetcher(BaseFetcher):
    """
    Akshare 数据源实现 - 集成 efinance 名称修复
    """
    name = "AkshareFetcher"
    priority = 1
    
    def __init__(self, sleep_min: float = 2.0, sleep_max: float = 5.0):
        super().__init__()
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self._last_request_time: Optional[float] = None
        self._name_map_cache: Dict[str, str] = {}  # 用于存储已修复的 A 股名称

    def _get_fixed_name(self, stock_code: str, ak_name: str = "") -> str:
        """
        专项修复：通过 efinance 纠正 A 股股票名称
        """
        # 1. 检查本地缓存
        if stock_code in self._name_map_cache:
            return self._name_map_cache[stock_code]
        
        # 2. 如果 Akshare 已经提供了合规名称，直接返回并存入缓存
        if ak_name and ak_name.strip() and ak_name.lower() != 'nan':
            self._name_map_cache[stock_code] = ak_name
            return ak_name

        # 3. 只有当名称缺失时，调用 efinance 修复 (A股为主)
        try:
            clean_code = re.sub(r'\D', '', stock_code)  # 提取 6 位数字代码
            base_info = ef.stock.get_base_info(clean_code)
            if not base_info.empty:
                # efinance 返回列名为 '股票名称'
                fixed_name = base_info.iloc[0]['股票名称']
                if fixed_name and str(fixed_name) != 'nan':
                    logger.info(f"[名称修复] 成功修复 {stock_code}: {fixed_name}")
                    self._name_map_cache[stock_code] = fixed_name
                    return fixed_name
        except Exception as e:
            logger.debug(f"efinance 修复 {stock_code} 名称失败: {e}")
            
        return ak_name

    def _set_random_user_agent(self) -> None:
        pass

    def _enforce_rate_limit(self) -> None:
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.sleep_min:
                time.sleep(self.sleep_min - elapsed)
        self.random_sleep(self.sleep_min, self.sleep_max)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if _is_hk_code(stock_code):
            return self._fetch_hk_data(stock_code, start_date, end_date)
        elif _is_etf_code(stock_code):
            return self._fetch_etf_data(stock_code, start_date, end_date)
        else:
            return self._fetch_stock_data(stock_code, start_date, end_date)

    # ----------------------------------------------------------------------
    # A 股实时行情获取与名称修复
    # ----------------------------------------------------------------------
    def _get_stock_realtime_quote(self, stock_code: str) -> Optional[RealtimeQuote]:
        import akshare as ak
        try:
            current_time = time.time()
            if (_realtime_cache['data'] is not None and 
                current_time - _realtime_cache['timestamp'] < _realtime_cache['ttl']):
                df = _realtime_cache['data']
            else:
                self._enforce_rate_limit()
                df = ak.stock_zh_a_spot_em()
                _realtime_cache['data'] = df
                _realtime_cache['timestamp'] = current_time

            row = df[df['代码'] == stock_code]
            if row.empty: return None
            row = row.iloc[0]
            
            # --- 执行修复逻辑 ---
            raw_name = str(row.get('名称', ''))
            final_name = self._get_fixed_name(stock_code, raw_name)
            
            def safe_float(val, default=0.0):
                try: return float(val) if not pd.isna(val) else default
                except: return default

            return RealtimeQuote(
                code=stock_code,
                name=final_name, # 使用修复后的名称
                price=safe_float(row.get('最新价')),
                change_pct=safe_float(row.get('涨跌幅')),
                change_amount=safe_float(row.get('涨跌额')),
                volume_ratio=safe_float(row.get('量比')),
                turnover_rate=safe_float(row.get('换手率')),
                amplitude=safe_float(row.get('振幅')),
                pe_ratio=safe_float(row.get('市盈率-动态')),
                pb_ratio=safe_float(row.get('市净率')),
                total_mv=safe_float(row.get('总市值')),
                circ_mv=safe_float(row.get('流通市值')),
                change_60d=safe_float(row.get('60日涨跌幅')),
                high_52w=safe_float(row.get('52周最高')),
                low_52w=safe_float(row.get('52周最低')),
            )
        except Exception as e:
            logger.error(f"获取 A 股 {stock_code} 实时行情失败: {e}")
            return None

    # (此处省略 _fetch_stock_data, _fetch_etf_data, _fetch_hk_data, _normalize_data)
    # 保持你原文代码中的这些方法不变即可。

    def get_realtime_quote(self, stock_code: str) -> Optional[RealtimeQuote]:
        if _is_hk_code(stock_code):
            return self._get_hk_realtime_quote(stock_code)
        elif _is_etf_code(stock_code):
            return self._get_etf_realtime_quote(stock_code)
        else:
            return self._get_stock_realtime_quote(stock_code)

    def get_chip_distribution(self, stock_code: str) -> Optional[ChipDistribution]:
        # (保持原逻辑不变)
        import akshare as ak
        if _is_etf_code(stock_code): return None
        try:
            self._enforce_rate_limit()
            df = ak.stock_cyq_em(symbol=stock_code)
            if df.empty: return None
            latest = df.iloc[-1]
            def safe_float(val, default=0.0):
                try: return float(val) if not pd.isna(val) else default
                except: return default
            return ChipDistribution(
                code=stock_code,
                date=str(latest.get('日期', '')),
                profit_ratio=safe_float(latest.get('获利比例')),
                avg_cost=safe_float(latest.get('平均成本')),
                cost_90_low=safe_float(latest.get('90成本-低')),
                cost_90_high=safe_float(latest.get('90成本-高')),
                concentration_90=safe_float(latest.get('90集中度')),
                cost_70_low=safe_float(latest.get('70成本-低')),
                cost_70_high=safe_float(latest.get('70成本-高')),
                concentration_70=safe_float(latest.get('70集中度')),
            )
        except Exception as e:
            logger.error(f"获取 {stock_code} 筹码分布失败: {e}")
            return None

    def get_enhanced_data(self, stock_code: str, days: int = 60) -> Dict[str, Any]:
        result = {'code': stock_code, 'daily_data': None, 'realtime_quote': None, 'chip_distribution': None}
        try:
            result['daily_data'] = self.get_daily_data(stock_code, days=days)
        except Exception as e:
            logger.error(f"获取 {stock_code} 日线数据失败: {e}")
        result['realtime_quote'] = self.get_realtime_quote(stock_code)
        result['chip_distribution'] = self.get_chip_distribution(stock_code)
        return result


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    fetcher = AkshareFetcher()
    
    # 测试 A 股名称修复逻辑
    print("--- 正在测试 A 股名称修复 ---")
    res = fetcher.get_realtime_quote('600519')
    if res:
        print(f"代码: {res.code}, 名称: {res.name}, 价格: {res.price}")
    else:
        print("获取失败")
