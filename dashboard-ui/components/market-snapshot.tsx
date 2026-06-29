import { cn, fmtChange } from '@/lib/utils'
import type { MarketTick } from '@/lib/types'

interface Props { ticks: MarketTick[] }

export function MarketSnapshot({ ticks }: Props) {
  if (ticks.length === 0) {
    return (
      <p className="text-sm text-muted italic">
        No bar data yet — connect and start trading
      </p>
    )
  }

  return (
    <div className="flex flex-wrap gap-3">
      {ticks.map((t) => {
        const positive = t.change_pct >= 0
        return (
          <div
            key={t.symbol}
            className="card p-4 min-w-[120px] flex flex-col gap-1"
          >
            <span className="text-[11px] font-semibold uppercase tracking-widest text-muted">
              {t.symbol}
            </span>
            <span className="text-xl font-bold text-zinc-100 leading-none">
              ${t.last_price.toFixed(2)}
            </span>
            <span
              className={cn(
                'text-xs font-semibold',
                positive ? 'text-positive' : 'text-negative',
              )}
            >
              {t.change_pct != null ? fmtChange(t.change_pct) : '—'}
            </span>
          </div>
        )
      })}
    </div>
  )
}
