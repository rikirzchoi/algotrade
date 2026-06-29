'use client'
import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { EquityChart } from '@/components/equity-chart'
import { CandleChart } from '@/components/candle-chart'
import { StatCard } from '@/components/stat-card'
import { cn, fmt$$, fmtPct, pnlColor } from '@/lib/utils'
import type { EquityPoint, PerformanceRow, OhlcBar } from '@/lib/types'

export default function PerformancePage() {
  const [equity, setEquity]       = useState<EquityPoint[]>([])
  const [perf, setPerf]           = useState<PerformanceRow[]>([])
  const [gldBars, setGldBars]     = useState<OhlcBar[]>([])
  const [usoBars, setUsoBars]     = useState<OhlcBar[]>([])

  useEffect(() => {
    api.equityCurve().then(setEquity).catch(() => {})
    api.performance().then(setPerf).catch(() => {})
    api.bars('GLD', '4 hours', 100).then(setGldBars).catch(() => {})
    api.bars('USO', '4 hours', 100).then(setUsoBars).catch(() => {})
    const id = setInterval(() => {
      api.equityCurve().then(setEquity).catch(() => {})
      api.performance().then(setPerf).catch(() => {})
    }, 30_000)
    return () => clearInterval(id)
  }, [])

  const totalPnl  = perf.reduce((s, r) => s + r.total_pnl, 0)
  const avgSharpe = perf.length > 0 ? perf.reduce((s, r) => s + r.sharpe, 0) / perf.length : 0
  const minDD     = perf.length > 0 ? Math.min(...perf.map((r) => r.max_drawdown)) : 0
  const avgWR     = perf.length > 0 ? perf.reduce((s, r) => s + r.win_rate, 0) / perf.length : 0

  // Drawdown series from equity curve
  const drawdownData: EquityPoint[] = (() => {
    if (equity.length === 0) return []
    let peak = -Infinity
    return equity.map((pt) => {
      if (pt.cumulative_pnl > peak) peak = pt.cumulative_pnl
      return { ...pt, cumulative_pnl: pt.cumulative_pnl - peak }
    })
  })()

  return (
    <div className="flex flex-col gap-6 p-8">
      <div>
        <h1 className="text-2xl font-bold text-zinc-100 tracking-tight">Performance</h1>
        <p className="text-sm text-muted mt-0.5">All-time strategy metrics</p>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total PnL"
          accent="green"
          value={<span className={pnlColor(totalPnl)}>{fmt$$(totalPnl, true)}</span>}
        />
        <StatCard label="Sharpe" accent="orange" value={avgSharpe.toFixed(2)} />
        <StatCard
          label="Max Drawdown"
          accent="red"
          value={<span className="text-negative">{fmt$$(minDD)}</span>}
        />
        <StatCard
          label="Win Rate"
          accent="zinc"
          value={`${(avgWR * 100).toFixed(1)}%`}
        />
      </div>

      {/* Equity + Drawdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card p-5">
          <h2 className="text-[11px] font-semibold uppercase tracking-widest text-muted mb-4">
            Equity Curve (All Time)
          </h2>
          <EquityChart data={equity} height={260} />
        </div>
        <div className="card p-5">
          <h2 className="text-[11px] font-semibold uppercase tracking-widest text-muted mb-4">
            Drawdown
          </h2>
          <EquityChart data={drawdownData} height={260} />
        </div>
      </div>

      {/* Commodity charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card p-5">
          <h2 className="text-[11px] font-semibold uppercase tracking-widest text-muted mb-4">
            GLD — 4H Chart
          </h2>
          <CandleChart bars={gldBars} height={260} />
        </div>
        <div className="card p-5">
          <h2 className="text-[11px] font-semibold uppercase tracking-widest text-muted mb-4">
            USO — 4H Chart
          </h2>
          <CandleChart bars={usoBars} height={260} />
        </div>
      </div>

      {/* Per-strategy breakdown */}
      <div className="card p-5">
        <h2 className="text-[11px] font-semibold uppercase tracking-widest text-muted mb-4">
          Per-Strategy Breakdown
        </h2>
        {perf.length === 0 ? (
          <p className="text-sm text-muted italic">No trade data yet</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th className="text-right">Total PnL</th>
                  <th className="text-right">Trades</th>
                  <th className="text-right">Win Rate</th>
                  <th className="text-right">Sharpe</th>
                  <th className="text-right">Max DD</th>
                </tr>
              </thead>
              <tbody>
                {perf.map((row) => (
                  <tr key={row.strategy_id}>
                    <td className="font-semibold">{row.strategy_id}</td>
                    <td className={cn('text-right font-mono', pnlColor(row.total_pnl))}>
                      {fmt$$(row.total_pnl, true)}
                    </td>
                    <td className="text-right font-mono">{row.trade_count}</td>
                    <td className="text-right font-mono">
                      {(row.win_rate * 100).toFixed(1)}%
                    </td>
                    <td className="text-right font-mono">{row.sharpe.toFixed(2)}</td>
                    <td className="text-right font-mono text-negative">
                      {fmt$$(row.max_drawdown)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
