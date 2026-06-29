'use client'
import { useState } from 'react'
import { cn } from '@/lib/utils'
import { api } from '@/lib/api'
import type { Fill } from '@/lib/types'

interface Props {
  strategies: string[]
  activeStrategies: string[]
  fillsToday: Fill[]
}

export function StrategyPanel({ strategies, activeStrategies, fillsToday }: Props) {
  const [pending, setPending] = useState<string | null>(null)
  const activeSet = new Set(activeStrategies)

  async function toggle(id: string) {
    setPending(id)
    try { await api.toggleStrategy(id) } catch {}
    finally { setPending(null) }
  }

  if (strategies.length === 0) {
    return <p className="text-sm text-muted italic">No strategies registered</p>
  }

  return (
    <div className="flex flex-col gap-2">
      {strategies.map((sid) => {
        const active = activeSet.has(sid)
        const fills = fillsToday.filter((f) => f.strategy_id === sid).length
        const loading = pending === sid
        return (
          <div
            key={sid}
            className="card px-4 py-3 flex items-center gap-4"
          >
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-zinc-200 truncate">{sid}</p>
              <p className="text-xs text-muted mt-0.5">Fills today: {fills}</p>
            </div>

            <span
              className={cn(
                'badge text-[10px] shrink-0',
                active ? 'badge-green' : 'badge-red',
              )}
            >
              {active ? 'Active' : 'Paused'}
            </span>

            <button
              onClick={() => toggle(sid)}
              disabled={loading}
              className={cn(
                'px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border shrink-0',
                'border-border text-muted hover:text-zinc-200 hover:border-zinc-600',
                loading && 'opacity-50 cursor-not-allowed',
              )}
            >
              {loading ? '…' : active ? 'Pause' : 'Resume'}
            </button>
          </div>
        )
      })}
    </div>
  )
}
