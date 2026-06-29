import type { Metadata } from 'next'
import './globals.css'
import { Sidebar } from '@/components/sidebar'

export const metadata: Metadata = {
  title: 'AlgoTrade',
  description: 'Algorithmic trading dashboard',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background antialiased">
        <Sidebar />
        <main className="ml-[220px] min-h-screen flex flex-col">
          {children}
        </main>
      </body>
    </html>
  )
}
