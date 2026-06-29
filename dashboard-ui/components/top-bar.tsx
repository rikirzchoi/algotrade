'use client'
import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

interface Props {
  title: string
  subtitle?: React.ReactNode
  connected?: boolean
  running?: boolean
  halted?: boolean
}

/** Glassy sticky page header with connection pill + live ET clock. */
export function TopBar({ title, subtitle, connected, running, halted }: Props) {
  const [now, setNow] = useState('')

  useEffect(() => {
    const tick = () =>
      setNow(
        new Date().toLocaleTimeString('en-US', {
          timeZone: 'America/New_York',
          hour12: false,
        }),
      )
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="glass px-8 py-4 flex items-center justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-xl font-bold tracking-tight text-zinc-100">{title}</h1>
        {subtitle && <p className="text-sm text-muted mt-0.5 truncate">{subtitle}</p>}
      </div>

      <div className="flex items-center gap-3 shrink-0">
        {halted && <span className="badge badge-red">HALTED</span>}

        <div className="flex items-center gap-2 rounded-full border border-white/[0.06] bg-surface/60 px-3 py-1.5">
          <span
            className={cn(
              'w-2 h-2 rounded-full',
              connected ? 'bg-positive animate-pulse-ring' : 'bg-negative',
            )}
          />
          <span className="text-xs font-medium text-zinc-300">
            {connected ? (running ? 'Live' : 'Connected') : 'Offline'}
          </span>
        </div>

        <div className="hidden sm:flex items-center gap-1.5 text-xs text-muted">
          <span className="font-mono tnum text-zinc-400">{now}</span>
          <span className="text-zinc-600">ET</span>
        </div>
      </div>
    </header>
  )
}
