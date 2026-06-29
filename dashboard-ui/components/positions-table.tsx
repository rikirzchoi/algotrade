import { cn } from '@/lib/utils'

interface Props { positions: Record<string, number> }

export function PositionsTable({ positions }: Props) {
  const entries = Object.entries(positions).filter(([, qty]) => qty !== 0)

  if (entries.length === 0) {
    return (
      <p className="text-sm text-muted italic py-4 text-center">
        No open positions
      </p>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="data-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Side</th>
            <th className="text-right">Qty</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([symbol, qty]) => {
            const isLong = qty > 0
            return (
              <tr key={symbol}>
                <td className="font-semibold">{symbol}</td>
                <td>
                  <span
                    className={cn('badge text-[10px]', isLong ? 'badge-green' : 'badge-red')}
                  >
                    {isLong ? 'LONG' : 'SHORT'}
                  </span>
                </td>
                <td className="text-right font-mono">{Math.abs(qty)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
