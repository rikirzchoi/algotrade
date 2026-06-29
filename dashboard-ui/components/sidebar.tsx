'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import { BarChart2, Activity, Heart, Zap } from 'lucide-react'
import { cn } from '@/lib/utils'
import { KillSwitchDialog } from './kill-switch-dialog'
import { useEngineState } from '@/lib/use-engine-state'

const NAV = [
  { href: '/',            label: 'Overview',    icon: BarChart2 },
  { href: '/performance', label: 'Performance', icon: Activity  },
  { href: '/health',      label: 'Health',      icon: Heart     },
]

export function Sidebar() {
  const pathname = usePathname()
  const { state, wsConnected } = useEngineState()

  const isHalted    = state?.is_halted ?? false
  const isPaper     = true // determined by IBKR port; hardcoded here as safe default
  const dailyPnl    = state?.daily_pnl ?? 0
  const drawdownPct = (state?.drawdown_pct ?? 0) * 100

  return (
    <aside className="fixed inset-y-0 left-0 w-[220px] flex flex-col bg-[#0d0d0f] border-r border-white/[0.06] z-40">
      {/* Logo */}
      <div className="px-5 pt-6 pb-4">
        <div className="flex items-center gap-2">
          <span className="grid place-items-center w-7 h-7 rounded-lg bg-accent/15 ring-1 ring-accent/30">
            <Zap className="w-4 h-4 text-accent" fill="currentColor" />
          </span>
          <span className="text-base font-bold text-zinc-100 tracking-tight">AlgoTrade</span>
        </div>
        <div className="mt-2 flex items-center gap-1.5">
          <span
            className={cn(
              'w-1.5 h-1.5 rounded-full',
              wsConnected ? 'bg-positive animate-pulse-ring' : 'bg-negative',
            )}
          />
          <span className="text-[11px] text-muted font-medium">
            {wsConnected ? 'Live' : 'Disconnected'}
          </span>
          {isHalted && <span className="badge badge-red text-[9px] ml-1">HALTED</span>}
        </div>
      </div>

      <div className="h-px bg-white/[0.06] mx-3" />

      {/* Quick stats */}
      <div className="px-4 py-3 grid grid-cols-2 gap-2">
        <div className="card p-2.5">
          <p className="text-[10px] text-muted uppercase tracking-wider mb-0.5">PnL</p>
          <p className={cn('text-sm font-bold leading-none tnum', dailyPnl >= 0 ? 'text-positive' : 'text-negative')}>
            {dailyPnl >= 0 ? '+' : ''}${Math.abs(dailyPnl).toFixed(0)}
          </p>
        </div>
        <div className="card p-2.5">
          <p className="text-[10px] text-muted uppercase tracking-wider mb-0.5">DD</p>
          <p
            className={cn(
              'text-sm font-bold leading-none tnum',
              drawdownPct > 3 ? 'text-negative' : drawdownPct > 1 ? 'text-warning' : 'text-zinc-400',
            )}
          >
            {drawdownPct.toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="h-px bg-white/[0.06] mx-3" />

      {/* Nav */}
      <nav className="flex-1 px-3 py-3 flex flex-col gap-0.5 overflow-y-auto">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-zinc-600 px-3 mb-1">
          Navigation
        </p>
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === '/' ? pathname === '/' : pathname.startsWith(href)
          return (
            <Link key={href} href={href} className={cn('nav-item', active && 'active')}>
              {active && (
                <>
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-lg bg-white/[0.06] border border-white/[0.08]"
                    transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                  />
                  <motion.span
                    layoutId="nav-accent"
                    className="absolute left-0 top-1/2 -translate-y-1/2 h-5 w-0.5 rounded-full bg-accent"
                    transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                  />
                </>
              )}
              <Icon className="w-4 h-4 shrink-0 relative z-10" />
              <span className="relative z-10">{label}</span>
            </Link>
          )
        })}
      </nav>

      {/* Bottom */}
      <div className="px-3 pb-5 pt-3 border-t border-white/[0.06] space-y-2">
        <div className="flex items-center justify-between px-1 mb-1">
          <span className="text-[11px] text-muted">{isPaper ? 'Paper Trading' : 'Live Trading'}</span>
          <span className={cn('w-2 h-2 rounded-full', isPaper ? 'bg-positive' : 'bg-negative')} />
        </div>
        <KillSwitchDialog isHalted={isHalted} />
      </div>
    </aside>
  )
}
