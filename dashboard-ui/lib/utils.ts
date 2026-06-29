import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function fmt$$(n: number, sign = false): string {
  const abs = Math.abs(n)
  const prefix = sign ? (n >= 0 ? '+' : '-') : n < 0 ? '-' : ''
  if (abs >= 1_000_000) return `${prefix}$${(abs / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000)     return `${prefix}$${(abs / 1_000).toFixed(1)}k`
  return `${prefix}$${abs.toFixed(2)}`
}

export function fmtPct(n: number, sign = false): string {
  return `${sign && n > 0 ? '+' : ''}${(n * 100).toFixed(2)}%`
}

export function fmtChange(pct: number): string {
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`
}

export function pnlColor(n: number): string {
  if (n > 0) return 'text-positive'
  if (n < 0) return 'text-negative'
  return 'text-muted'
}

export function toTimestamp(dateStr: string): number {
  return Math.floor(new Date(dateStr).getTime() / 1000)
}
