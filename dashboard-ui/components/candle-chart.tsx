'use client'
import { useEffect, useRef } from 'react'
import { createChart, ColorType } from 'lightweight-charts'
import type { OhlcBar } from '@/lib/types'
import { toTimestamp } from '@/lib/utils'

interface Props {
  bars: OhlcBar[]
  height?: number
}

export function CandleChart({ bars, height = 280 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current || bars.length === 0) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#71717a',
        fontFamily: 'Inter, system-ui, sans-serif',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.04)' },
        horzLines: { color: 'rgba(255,255,255,0.04)' },
      },
      crosshair: {
        vertLine: { color: 'rgba(249,115,22,0.4)', labelBackgroundColor: '#f97316' },
        horzLine: { color: 'rgba(249,115,22,0.4)', labelBackgroundColor: '#f97316' },
      },
      rightPriceScale: { borderColor: 'rgba(255,255,255,0.06)' },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.06)',
        timeVisible: true,
        secondsVisible: false,
      },
      width: containerRef.current.clientWidth,
      height,
    })

    const candleSeries = chart.addCandlestickSeries({
      upColor:          '#22c55e',
      downColor:        '#ef4444',
      borderUpColor:    '#22c55e',
      borderDownColor:  '#ef4444',
      wickUpColor:      '#22c55e',
      wickDownColor:    '#ef4444',
    })

    const seen = new Set<number>()
    const sorted = [...bars]
      .sort((a, b) => toTimestamp(a.timestamp) - toTimestamp(b.timestamp))
      .filter((b) => {
        const t = toTimestamp(b.timestamp)
        if (seen.has(t)) return false
        seen.add(t)
        return true
      })
    candleSeries.setData(
      sorted.map((b) => ({
        time:  toTimestamp(b.timestamp) as any,
        open:  b.open,
        high:  b.high,
        low:   b.low,
        close: b.close,
      })),
    )

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
  }, [bars, height])

  return (
    <div className="relative w-full">
      {bars.length === 0 ? (
        <div
          style={{ height }}
          className="flex items-center justify-center text-muted text-sm"
        >
          No bar data
        </div>
      ) : (
        <div ref={containerRef} className="w-full tv-chart" />
      )}
    </div>
  )
}
