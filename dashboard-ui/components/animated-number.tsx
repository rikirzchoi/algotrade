'use client'
import { useEffect } from 'react'
import { motion, useSpring, useTransform } from 'framer-motion'

interface Props {
  value: number
  format: (n: number) => string
  className?: string
}

/** Smoothly tweens between numeric values with tabular figures (no layout jiggle). */
export function AnimatedNumber({ value, format, className }: Props) {
  const spring = useSpring(value, { stiffness: 140, damping: 22, mass: 0.7 })
  const text = useTransform(spring, (v) => format(v))

  useEffect(() => {
    spring.set(value)
  }, [value, spring])

  return <motion.span className={className}>{text}</motion.span>
}
