'use client'
import { useEffect, useRef, useState } from 'react'
import type { EngineState } from './types'

const WS_URL =
  typeof window !== 'undefined'
    ? `ws://${window.location.hostname}:8000/ws`
    : 'ws://localhost:8000/ws'

export function useEngineState() {
  const [state, setState] = useState<EngineState | null>(null)
  const [wsConnected, setWsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    function connect() {
      if (wsRef.current?.readyState === WebSocket.OPEN) return
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        setWsConnected(true)
        if (retryRef.current) clearTimeout(retryRef.current)
      }
      ws.onmessage = (e) => {
        try { setState(JSON.parse(e.data as string) as EngineState) } catch {}
      }
      ws.onclose = () => {
        setWsConnected(false)
        retryRef.current = setTimeout(connect, 2_000)
      }
      ws.onerror = () => ws.close()
    }

    connect()
    return () => {
      if (retryRef.current) clearTimeout(retryRef.current)
      wsRef.current?.close()
    }
  }, [])

  return { state, wsConnected }
}
