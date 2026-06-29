export interface Fill {
  strategy_id: string
  symbol: string
  direction: 'LONG' | 'SHORT' | 'FLAT'
  quantity: number
  fill_price: number
  commission: number
  timestamp: string
}

export interface EngineState {
  connected: boolean
  is_running: boolean
  is_halted: boolean
  daily_pnl: number
  peak_equity: number
  drawdown_pct: number
  positions: Record<string, number>
  active_strategies: string[]
  fills_today: Fill[]
  error_count_today: number
  last_heartbeat: string | null
  data_stale: boolean
}

export interface EquityPoint {
  strategy_id: string
  symbol: string
  exit_time: string
  pnl: number
  cumulative_pnl: number
}

export interface OhlcBar {
  symbol: string
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  bar_size: string
}

export interface PerformanceRow {
  strategy_id: string
  total_pnl: number
  trade_count: number
  win_rate: number
  sharpe: number
  max_drawdown: number
}

export interface SystemEvent {
  timestamp: string
  kind: string
  message: string
}

export interface MarketTick {
  symbol: string
  last_price: number
  last_ts: string
  day_open: number
  change_pct: number
}
