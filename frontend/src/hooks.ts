import { DependencyList, useEffect, useRef } from 'react'

export function usePolling(
  callback: () => void | Promise<void>,
  intervalMs: number,
  deps: DependencyList = [],
  enabled: boolean = true,
) {
  const callbackRef = useRef(callback)

  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  useEffect(() => {
    if (!enabled) return
    void callbackRef.current()
  }, [enabled, ...deps])

  useEffect(() => {
    if (!enabled) return
    const timer = window.setInterval(() => {
      void callbackRef.current()
    }, intervalMs)
    return () => window.clearInterval(timer)
  }, [intervalMs, enabled, ...deps])
}
