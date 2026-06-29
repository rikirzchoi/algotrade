'use client'
import { useState } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { TriangleAlert } from 'lucide-react'
import { cn } from '@/lib/utils'
import { api } from '@/lib/api'

interface Props { isHalted: boolean }

export function KillSwitchDialog({ isHalted }: Props) {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)

  async function handleConfirm() {
    setLoading(true)
    try { await api.killSwitch() } catch {}
    finally { setLoading(false); setOpen(false) }
  }

  async function handleResume() {
    setLoading(true)
    try { await api.resume() } catch {}
    finally { setLoading(false) }
  }

  return (
    <div className="flex flex-col gap-2 w-full">
      {isHalted && (
        <button
          onClick={handleResume}
          disabled={loading}
          className="w-full py-2 px-3 rounded-lg bg-warning/10 border border-warning/30
                     text-warning text-xs font-semibold hover:bg-warning/20 transition-colors"
        >
          {loading ? 'Resuming…' : 'Resume Trading'}
        </button>
      )}

      <Dialog.Root open={open} onOpenChange={setOpen}>
        <Dialog.Trigger asChild>
          <button
            className="w-full py-2 px-3 rounded-lg bg-transparent border border-negative/40
                       text-negative text-xs font-bold uppercase tracking-wider
                       hover:bg-negative hover:text-white transition-all"
          >
            ■ Emergency Stop
          </button>
        </Dialog.Trigger>

        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50" />
          <Dialog.Content
            className={cn(
              'fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50',
              'card p-6 w-[360px] shadow-2xl',
            )}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-negative/10 flex items-center justify-center shrink-0">
                <TriangleAlert className="w-5 h-5 text-negative" />
              </div>
              <div>
                <Dialog.Title className="text-base font-bold text-zinc-100">
                  Emergency Stop
                </Dialog.Title>
                <Dialog.Description className="text-xs text-muted mt-0.5">
                  This cancels all open orders and halts trading immediately.
                </Dialog.Description>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <Dialog.Close asChild>
                <button className="flex-1 py-2 rounded-lg border border-border text-muted
                                   text-sm hover:text-zinc-200 hover:border-zinc-600 transition-colors">
                  Cancel
                </button>
              </Dialog.Close>
              <button
                onClick={handleConfirm}
                disabled={loading}
                className="flex-1 py-2 rounded-lg bg-negative text-white text-sm font-bold
                           hover:bg-red-700 transition-colors disabled:opacity-50"
              >
                {loading ? 'Stopping…' : 'Confirm Stop'}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  )
}
