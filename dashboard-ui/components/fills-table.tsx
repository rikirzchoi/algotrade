import { cn } from '@/lib/utils'
import type { Fill } from '@/lib/types'

interface Props { fills: Fill[] }

export function FillsTable({ fills }: Props) {
  if (fills.length === 0) {
    return (
      <p className="text-sm text-muted italic py-4 text-center">No fills today</p>
    )
  }

  const sorted = [...fills].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
  )

  return (
    <div className="overflow-x-auto">
      <table className="data-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Strategy</th>
            <th>Symbol</th>
            <th>Side</th>
            <th className="text-right">Qty</th>
            <th className="text-right">Price</th>
            <th className="text-right">Comm.</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((f, i) => {
            const isLong = f.direction === 'LONG'
            const t = new Date(f.timestamp)
            const timeStr = isNaN(t.getTime())
              ? f.timestamp
              : t.toLocaleTimeString('en-US', { hour12: false })
            return (
              <tr key={i}>
                <td className="font-mono text-xs text-muted">{timeStr}</td>
                <td className="text-xs">{f.strategy_id}</td>
                <td className="font-semibold">{f.symbol}</td>
                <td>
                  <span
                    className={cn(
                      'badge text-[10px]',
                      isLong ? 'badge-green' : 'badge-red',
                    )}
                  >
                    {f.direction}
                  </span>
                </td>
                <td className="text-right font-mono">{f.quantity}</td>
                <td className="text-right font-mono">${f.fill_price.toFixed(2)}</td>
                <td className="text-right font-mono text-muted">
                  ${f.commission.toFixed(2)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
