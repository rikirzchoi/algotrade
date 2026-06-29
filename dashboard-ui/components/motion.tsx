'use client'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

const EASE: [number, number, number, number] = [0.16, 1, 0.3, 1]

/**
 * Reveal-on-mount wrapper. Each element animates in independently and settles at
 * opacity 1 — re-renders (WS/poll updates) never reset it, so content can't get
 * stuck hidden. `delay` provides a manual stagger.
 */
export function Section({
  children,
  className,
  delay = 0,
}: {
  children: React.ReactNode
  className?: string
  delay?: number
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: EASE, delay }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('skeleton', className)} />
}
