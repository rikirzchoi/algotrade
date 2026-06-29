import type {
  EngineState, EquityPoint, OhlcBar,
  PerformanceRow, SystemEvent, MarketTick,
} from './types'

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`${path}: ${res.status}`)
  return res.json() as Promise<T>
}

async function post(path: string): Promise<{ ok: boolean }> {
  const res = await fetch(`${BASE}${path}`, { method: 'POST' })
  return res.json()
}

export const api = {
  state:          ()                    => get<EngineState>('/state'),
  strategies:     ()                    => get<string[]>('/strategies'),
  equityCurve:    ()                    => get<EquityPoint[]>('/equity-curve'),
  bars:           (sym: string, size = '1 day', limit = 200) =>
                    get<OhlcBar[]>(`/bars/${sym}?bar_size=${encodeURIComponent(size)}&limit=${limit}`),
  performance:    ()                    => get<PerformanceRow[]>('/performance'),
  systemEvents:   (limit = 50)          => get<SystemEvent[]>(`/system-events?limit=${limit}`),
  marketSnapshot: ()                    => get<MarketTick[]>('/market-snapshot'),
  toggleStrategy: (id: string)          => post(`/strategies/${id}/toggle`),
  killSwitch:     ()                    => post('/kill-switch'),
  resume:         ()                    => post('/resume'),
}
