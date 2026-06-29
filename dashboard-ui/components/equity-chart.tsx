'use client'
import { useEffect, useRef } from 'react'
import { createChart, ColorType, LineStyle } from 'lightweight-charts'
import type { EquityPoint } from '@/lib/types'
import { toTimestamp } from '@/lib/utils'

interface Props {
  data: EquityPoint[]
  height?: number
}

const COLORS = ['#f97316', '#3b82f6', '#22c55e', '#ef4444', '#a855f7']

export function EquityChart({ data, height = 300 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#71717a',
        fontFamily: 'Inter, system-ui, sans-serif',
        fontSize: 12,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.04)' },
        horzLines: { color: 'rgba(255,255,255,0.04)' },
      },
      crosshair: {
        vertLine: {
          color: 'rgba(249,115,22,0.4)',
          labelBackgroundColor: '#f97316',
        },
        horzLine: {
          color: 'rgba(249,115,22,0.4)',
          labelBackgroundColor: '#f97316',
        },
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.06)',
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.06)',
        timeVisible: true,
        secondsVisible: false,
      },
      width: containerRef.current.clientWidth,
      height,
    })

    // Group by strategy_id
    const byStrategy = new Map<string, EquityPoint[]>()
    for (const pt of data) {
      const arr = byStrategy.get(pt.strategy_id) ?? []
      arr.push(pt)
      byStrategy.set(pt.strategy_id, arr)
    }

    let colorIdx = 0
    for (const [, points] of byStrategy) {
      const series = chart.addLineSeries({
        color: COLORS[colorIdx++ % COLORS.length],
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 4,
      })
      const sorted = [...points].sort(
        (a, b) => toTimestamp(a.exit_time) - toTimestamp(b.exit_time),
      )
      series.setData(
        sorted.map((p) => ({
          time: toTimestamp(p.exit_time) as any,
          value: p.cumulative_pnl,
        })),
      )
    }

    chart.timeScale().fitContent()

    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width
      if (w) chart.applyOptions({ width: w })
    })
    observer.observe(containerRef.current)

    return () => {
      observer.disconnect()
      chart.remove()
    }
  }, [data, height])

  return (
    <div className="relative w-full">
      {data.length === 0 ? (
        <div
          style={{ height }}
          className="flex items-center justify-center text-muted text-sm"
        >
          No trade data yet
        </div>
      ) : (
        <div ref={containerRef} className="w-full tv-chart" />
      )}
    </div>
  )
}
