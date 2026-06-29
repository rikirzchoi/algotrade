'use client'
import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { AnimatedNumber } from './animated-number'

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1]

type Accent = 'orange' | 'red' | 'green' | 'zinc'
type Tone = 'pos' | 'neg' | 'warn' | 'neutral'

interface Props {
  label: string
  /** A number (animated, with format) or any node (rendered as-is, legacy). */
  value: number | React.ReactNode
  format?: (n: number) => string
  accent?: Accent
  tone?: Tone
  sub?: React.ReactNode
  icon?: LucideIcon
  delay?: number
}

const toneClass: Record<Tone, string> = {
  pos: 'text-positive',
  neg: 'text-negative',
  warn: 'text-warning',
  neutral: 'text-zinc-100',
}

export function StatCard({ label, value, format, accent = 'zinc', tone = 'neutral', sub, icon: Icon, delay = 0 }: Props) {
  const isNumeric = typeof value === 'number' && typeof format === 'function'
  const prev = useRef(typeof value === 'number' ? value : 0)
  const [flash, setFlash] = useState<'up' | 'down' | null>(null)

  useEffect(() => {
    if (typeof value !== 'number') return
    if (value > prev.current) setFlash('up')
    else if (value < prev.current) setFlash('down')
    prev.current = value
    const t = setTimeout(() => setFlash(null), 700)
    return () => clearTimeout(t)
  }, [value])

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: EASE, delay }}
      className="card card-hover p-5 flex flex-col gap-2 min-w-0 overflow-hidden"
    >
      <span className={cn('accent-bar', `accent-${accent}`)} />

      {/* flash wash on value change */}
      <AnimatePresence>
        {flash && (
          <motion.span
            key={Date.now()}
            initial={{ opacity: 0.6 }}
            animate={{ opacity: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.7 }}
            className={cn(
              'pointer-events-none absolute inset-0',
              flash === 'up' ? 'bg-positive/[0.07]' : 'bg-negative/[0.07]',
            )}
          />
        )}
      </AnimatePresence>

      <div className="flex items-center justify-between">
        <p className="eyebrow">{label}</p>
        {Icon && <Icon className="w-3.5 h-3.5 text-zinc-600" />}
      </div>

      {isNumeric ? (
        <AnimatedNumber
          value={value as number}
          format={format!}
          className={cn('text-3xl font-bold tracking-tight leading-none tnum', toneClass[tone])}
        />
      ) : (
        <p className="text-3xl font-bold tracking-tight leading-none tnum text-zinc-100">{value}</p>
      )}

      {sub && <p className="text-xs text-muted mt-0.5">{sub}</p>}
    </motion.div>
  )
}
