'use client'
import { useEffect, useState } from 'react'
import { Wifi, Server, Activity, AlertTriangle } from 'lucide-react'
import { useEngineState } from '@/lib/use-engine-state'
import { api } from '@/lib/api'
import { cn } from '@/lib/utils'
import { StatCard } from '@/components/stat-card'
import { TopBar } from '@/components/top-bar'
import { Section } from '@/components/motion'
import type { SystemEvent } from '@/lib/types'

export default function HealthPage() {
  const { state, wsConnected } = useEngineState()
  const [events, setEvents] = useState<SystemEvent[]>([])

  useEffect(() => {
    api.systemEvents(30).then(setEvents).catch(() => {})
    const id = setInterval(() => {
      api.systemEvents(30).then(setEvents).catch(() => {})
    }, 10_000)
    return () => clearInterval(id)
  }, [])

  const errorCount  = state?.error_count_today ?? 0
  const lastHb      = state?.last_heartbeat
  const isConnected = wsConnected

  const lastHbStr = (() => {
    if (!lastHb) return '—'
    const d = new Date(lastHb)
    if (isNaN(d.getTime())) return '—'
    const diffSec = Math.floor((Date.now() - d.getTime()) / 1000)
    if (diffSec < 60) return `${diffSec}s ago`
    return `${Math.floor(diffSec / 60)}m ago`
  })()

  return (
    <>
      <TopBar
        title="Health"
        subtitle="System status and events"
        connected={wsConnected}
        running={state?.is_running}
        halted={state?.is_halted}
      />

      <div className="flex flex-col gap-6 p-8">
        {/* Status cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Connection"
            accent={isConnected ? 'green' : 'red'}
            icon={Wifi}
            sub="WebSocket to engine"
            value={
              <span className={isConnected ? 'text-positive' : 'text-negative'}>
                {isConnected ? 'Live' : 'Off'}
              </span>
            }
          />
          <StatCard
            label="Engine"
            accent={state?.is_running ? 'green' : 'zinc'}
            icon={Server}
            value={
              <span className={state?.is_running ? 'text-positive' : 'text-muted'}>
                {state?.is_running ? 'Running' : 'Stopped'}
              </span>
            }
          />
          <StatCard label="Last Heartbeat" accent="zinc" icon={Activity} value={lastHbStr} />
          <StatCard
            label="Errors Today"
            accent={errorCount > 0 ? 'red' : 'zinc'}
            icon={AlertTriangle}
            value={errorCount}
            format={(n) => `${Math.round(n)}`}
            tone={errorCount > 0 ? 'neg' : 'neutral'}
          />
        </div>

        {/* Kill-switch state */}
        {state?.is_halted && (
          <Section className="card p-4 border-negative/30 bg-negative/5">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-negative animate-pulse" />
              <div>
                <p className="text-sm font-bold text-negative">Kill Switch Active</p>
                <p className="text-xs text-muted mt-0.5">
                  All new orders are blocked. Use the sidebar to resume.
                </p>
              </div>
            </div>
          </Section>
        )}

        {/* Market-data staleness */}
        {state?.data_stale && (
          <Section className="card p-4 border-warning/40 bg-warning/5">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-warning animate-pulse" />
              <div>
                <p className="text-sm font-bold text-warning">Market Data Stale</p>
                <p className="text-xs text-muted mt-0.5">
                  Engine is connected but no live bars are arriving — strategies are
                  blind and no orders will fire. Check the TWS market-data subscription
                  / data farm.
                </p>
              </div>
            </div>
          </Section>
        )}

        {/* System events */}
        <Section delay={0.06} className="card p-5">
          <h2 className="eyebrow mb-4">System Events</h2>
          {events.length === 0 ? (
            <p className="text-sm text-muted italic">No events logged</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Type</th>
                    <th>Message</th>
                  </tr>
                </thead>
                <tbody>
                  {[...events].reverse().map((evt, i) => {
                    const kind = String(evt.kind ?? '')
                    const isError   = kind === 'ERROR'
                    const isWarning = kind === 'WARNING'
                    return (
                      <tr key={i} className={cn(isError && 'bg-negative/5', isWarning && 'bg-warning/5')}>
                        <td className="font-mono text-xs text-muted whitespace-nowrap">
                          {String(evt.timestamp ?? '').slice(0, 19)}
                        </td>
                        <td>
                          <span
                            className={cn(
                              'badge text-[10px]',
                              isError && 'badge-red',
                              isWarning && 'badge-orange',
                              !isError && !isWarning && 'badge-zinc',
                            )}
                          >
                            {kind}
                          </span>
                        </td>
                        <td className="text-xs text-zinc-400 max-w-xs truncate">
                          {String(evt.message ?? '')}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Section>

        {/* Active strategies */}
        <Section delay={0.1} className="card p-5">
          <h2 className="eyebrow mb-4">Active Strategies</h2>
          {(state?.active_strategies ?? []).length === 0 ? (
            <p className="text-sm text-muted italic">None running</p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {(state?.active_strategies ?? []).map((s) => (
                <span key={s} className="badge badge-green">{s}</span>
              ))}
            </div>
          )}
        </Section>
      </div>
    </>
  )
}
