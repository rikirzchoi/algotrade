import { cn } from '@/lib/utils'

type Accent = 'orange' | 'red' | 'green' | 'zinc'

interface Props {
  label: string
  value: React.ReactNode
  sub?: React.ReactNode
  accent?: Accent
}

export function StatCard({ label, value, sub, accent = 'zinc' }: Props) {
  return (
    <div
      className={cn(
        'card stat-accent-left p-5 flex flex-col gap-2 min-w-0',
        `stat-accent-${accent}`,
      )}
    >
      <p className="text-[11px] font-semibold uppercase tracking-widest text-muted">
        {label}
      </p>
      <p className="text-3xl font-bold tracking-tight text-zinc-100 leading-none">
        {value}
      </p>
      {sub && <p className="text-xs text-muted">{sub}</p>}
    </div>
  )
}
