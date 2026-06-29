'use client'
import { useEffect, useState } from 'react'
import { useEngineState } from '@/lib/use-engine-state'
import { api } from '@/lib/api'
import { StatCard } from '@/components/stat-card'
import { EquityChart } from '@/components/equity-chart'
import { FillsTable } from '@/components/fills-table'
import { PositionsTable } from '@/components/positions-table'
import { StrategyPanel } from '@/components/strategy-panel'
import { MarketSnapshot } from '@/components/market-snapshot'
import { cn, fmt$$, fmtPct, pnlColor } from '@/lib/utils'
import type { EquityPoint, MarketTick } from '@/lib/types'

export default function OverviewPage() {
  const { state } = useEngineState()
  const [equityCurve, setEquityCurve] = useState<EquityPoint[]>([])
  const [strategies, setStrategies]   = useState<string[]>([])
  const [ticks, setTicks]             = useState<MarketTick[]>([])

  useEffect(() => {
    api.equityCurve().then(setEquityCurve).catch(() => {})
    api.strategies().then(setStrategies).catch(() => {})
    api.marketSnapshot().then(setTicks).catch(() => {})
    const id = setInterval(() => {
      api.equityCurve().then(setEquityCurve).catch(() => {})
      api.marketSnapshot().then(setTicks).catch(() => {})
    }, 5_000)
    return () => clearInterval(id)
  }, [])

  const dailyPnl    = state?.daily_pnl    ?? 0
  const drawdownPct = state?.drawdown_pct ?? 0
  const fills       = state?.fills_today  ?? []
  const positions   = state?.positions    ?? {}
  const active      = state?.active_strategies ?? []

  const winRate = fills.length > 0
    ? fills.filter((f) => f.direction === 'LONG').length / fills.length
    : null

  const totalPnl = equityCurve.reduce((sum, pt) => sum + pt.pnl, 0)

  return (
    <div className="flex flex-col gap-6 p-8">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-zinc-100 tracking-tight">Overview</h1>
        <p className="text-sm text-muted mt-0.5">
          {state?.is_running ? 'Engine running' : 'Engine stopped'}
          {state?.is_halted ? ' · HALTED' : ''}
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Daily PnL"
          accent="orange"
          value={
            <span className={pnlColor(dailyPnl)}>{fmt$$(dailyPnl, true)}</span>
          }
        />
        <StatCard
          label="Drawdown"
          accent="red"
          value={
            <span className={drawdownPct > 0.04 ? 'text-negative' : drawdownPct > 0.02 ? 'text-warning' : 'text-zinc-300'}>
              {fmtPct(drawdownPct)}
            </span>
          }
        />
        <StatCard
          label="Win Rate"
          accent="zinc"
          value={winRate != null ? `${(winRate * 100).toFixed(0)}%` : '—'}
          sub={`${fills.length} fills today`}
        />
        <StatCard
          label="Total PnL"
          accent="green"
          value={
            <span className={pnlColor(totalPnl)}>{fmt$$(totalPnl, true)}</span>
          }
        />
      </div>

      {/* Market snapshot */}
      <div className="card p-5">
        <h2 className="text-[11px] font-semibold uppercase tracking-widest text-muted mb-4">
          Market Snapshot
        </h2>
        <MarketSnapshot ticks={ticks} />
      </div>

      {/* Equity curve */}
      <div className="card p-5">
        <h2 className="text-[11px] font-semibold uppercase tracking-widest text-muted mb-4">
          Today&apos;s Equity Curve
        </h2>
        <EquityChart
          data={equityCurve.filter((p) => {
            const d = new Date(p.exit_time)
            return d.toDateString() === new Date().toDateString()
          })}
          height={280}
        />
      </div>

      {/* Fills + Positions */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <div className="card p-5 lg:col-span-3">
          <h2 className="text-[11px] font-semibold uppercase tracking-widest text-muted mb-4">
            Today&apos;s Fills
          </h2>
          <FillsTable fills={fills} />
        </div>
        <div className="card p-5 lg:col-span-2">
          <h2 className="text-[11px] font-semibold uppercase tracking-widest text-muted mb-4">
            Open Positions
          </h2>
          <PositionsTable positions={positions} />
        </div>
      </div>

      {/* Strategy controls */}
      <div className="card p-5">
        <h2 className="text-[11px] font-semibold uppercase tracking-widest text-muted mb-4">
          Strategy Controls
        </h2>
        <StrategyPanel
          strategies={strategies}
          activeStrategies={active}
          fillsToday={fills}
        />
      </div>
    </div>
  )
}
