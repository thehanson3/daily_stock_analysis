def get_chip_distribution(self, stock_code: str) -> Optional[ChipDistribution]:
    """
    获取筹码分布数据（基于官方文档修正）
    """
    print(f"[DEBUG] 开始获取 {stock_code} 的筹码分布数据")
    
    # ... [原有的过滤逻辑保持不变] ...
    
    try:
        # 数据更新时间为 19:00，在更新之后请求当天数据，否则请求前一个交易日数据
        china_now = self._get_china_now()
        current_hour = china_now.hour
        
        # 获取交易日（根据当前时间判断）
        trade_date = self.get_trade_time(early_time='00:00', late_time='19:00')
        # 如果当前时间晚于19点，请求当天数据；否则请求前一天数据
        if current_hour >= 19:
            trade_date = self.get_trade_time(early_time='19:01', late_time='23:59')
        else:
            trade_date = self.get_trade_time(early_time='00:00', late_time='18:59')
            
        print(f"[DEBUG] get_trade_time 返回: {trade_date}")
        if not trade_date:
            return None
        
        # ... [原有的股票代码转换逻辑保持不变] ...
        
        # -------- 以下是修改核心 --------
        # 优先使用 start_date 和 end_date
        print(f"[DEBUG] 调用 cyq_perf: ts_code={ts_code}, start_date={trade_date}, end_date={trade_date}")
        df = self._call_api_with_rate_limit(
            "cyq_perf",
            ts_code=ts_code,
            start_date=trade_date,
            end_date=trade_date,
            fields='ts_code,trade_date,winner_rate,weight_avg,cost_5pct,cost_15pct,cost_85pct,cost_95pct,concentration'
        )
        
        # 若 start_date+end_date 未返回数据，兼容仅使用 trade_date 参数
        if df is None or df.empty:
            print("[DEBUG] start_date+end_date 未返回数据，尝试仅使用 trade_date 参数")
            df = self._call_api_with_rate_limit(
                "cyq_perf",
                ts_code=ts_code,
                trade_date=trade_date,
                fields='ts_code,trade_date,winner_rate,weight_avg,cost_5pct,cost_15pct,cost_85pct,cost_95pct,concentration'
            )
        # -------- 修改结束 ----------
        
        print(f"[DEBUG] cyq_perf 返回 df: {type(df)}, 长度: {len(df) if df is not None else 'None'}")
        # ... [后续的数据处理和返回逻辑] ...