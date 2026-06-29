'use client'
import { useEffect, useState } from 'react'
import { DollarSign, TrendingDown, Target, Wallet } from 'lucide-react'
import { useEngineState } from '@/lib/use-engine-state'
import { api } from '@/lib/api'
import { StatCard } from '@/components/stat-card'
import { EquityChart } from '@/components/equity-chart'
import { FillsTable } from '@/components/fills-table'
import { PositionsTable } from '@/components/positions-table'
import { StrategyPanel } from '@/components/strategy-panel'
import { MarketSnapshot } from '@/components/market-snapshot'
import { TopBar } from '@/components/top-bar'
import { Section, Skeleton } from '@/components/motion'
import { fmt$$, fmtPct } from '@/lib/utils'
import type { EquityPoint, MarketTick } from '@/lib/types'

export default function OverviewPage() {
  const { state, wsConnected } = useEngineState()
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
    : 0

  const totalPnl = equityCurve.reduce((sum, pt) => sum + pt.pnl, 0)

  const todaysEquity = equityCurve.filter((p) => {
    const d = new Date(p.exit_time)
    return d.toDateString() === new Date().toDateString()
  })

  return (
    <>
      <TopBar
        title="Overview"
        subtitle={state?.is_running ? 'Engine running' : 'Engine stopped'}
        connected={wsConnected}
        running={state?.is_running}
        halted={state?.is_halted}
      />

      <div className="flex flex-col gap-6 p-8">
        {/* Stat cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Daily PnL"
            accent="orange"
            icon={DollarSign}
            value={dailyPnl}
            format={(n) => fmt$$(n, true)}
            tone={dailyPnl >= 0 ? 'pos' : 'neg'}
            delay={0}
          />
          <StatCard
            label="Drawdown"
            accent="red"
            icon={TrendingDown}
            value={drawdownPct}
            format={(n) => fmtPct(n)}
            tone={drawdownPct > 0.04 ? 'neg' : drawdownPct > 0.02 ? 'warn' : 'neutral'}
            delay={0.05}
          />
          <StatCard
            label="Win Rate"
            accent="zinc"
            icon={Target}
            value={winRate}
            format={(n) => (fills.length > 0 ? `${(n * 100).toFixed(0)}%` : '—')}
            sub={`${fills.length} fills today`}
            delay={0.1}
          />
          <StatCard
            label="Total PnL"
            accent="green"
            icon={Wallet}
            value={totalPnl}
            format={(n) => fmt$$(n, true)}
            tone={totalPnl >= 0 ? 'pos' : 'neg'}
            delay={0.15}
          />
        </div>

        {/* Market snapshot */}
        <Section delay={0.18} className="card p-5">
          <h2 className="eyebrow mb-4">Market Snapshot</h2>
          {ticks.length > 0 ? (
            <MarketSnapshot ticks={ticks} />
          ) : (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-16" />
              ))}
            </div>
          )}
        </Section>

        {/* Equity curve */}
        <Section delay={0.22} className="card p-5">
          <h2 className="eyebrow mb-4">Today&apos;s Equity Curve</h2>
          {todaysEquity.length > 0 ? (
            <EquityChart data={todaysEquity} height={280} />
          ) : (
            <Skeleton className="h-[280px]" />
          )}
        </Section>

        {/* Fills + Positions */}
        <Section delay={0.26} className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          <div className="card p-5 lg:col-span-3">
            <h2 className="eyebrow mb-4">Today&apos;s Fills</h2>
            <FillsTable fills={fills} />
          </div>
          <div className="card p-5 lg:col-span-2">
            <h2 className="eyebrow mb-4">Open Positions</h2>
            <PositionsTable positions={positions} />
          </div>
        </Section>

        {/* Strategy controls */}
        <Section delay={0.3} className="card p-5">
          <h2 className="eyebrow mb-4">Strategy Controls</h2>
          <StrategyPanel
            strategies={strategies}
            activeStrategies={active}
            fillsToday={fills}
          />
        </Section>
      </div>
    </>
  )
}
