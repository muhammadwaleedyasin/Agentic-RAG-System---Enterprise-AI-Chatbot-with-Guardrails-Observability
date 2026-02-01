'use client';

import { Inter } from 'next/font/google';
import { usePathname } from 'next/navigation';
import './globals.css';
import { Providers } from '@/components/providers';
import { Navigation } from '@/components/layout/navigation';
import { Header } from '@/components/layout/header';
import { metadata } from './metadata';

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <title>{metadata.title}</title>
        <meta name="description" content={metadata.description} />
        <meta name="viewport" content={metadata.viewport} />
        <meta name="robots" content={metadata.robots} />
      </head>
      <body className={`${inter.className} bg-white dark:bg-secondary-950 text-secondary-900 dark:text-secondary-100 antialiased`}>
        <a
          href="#main-content"
          className="skip-link focus:not-sr-only"
          aria-label="Skip to main content"
        >
          Skip to main content
        </a>
        
        <Providers>
          <div className="min-h-screen flex flex-col">
            {!pathname?.startsWith('/login') && <Header />}
            
            <div className="flex flex-1">
              {!pathname?.startsWith('/login') && <Navigation />}
              
              <main
                id="main-content"
                className="flex-1 p-4 md:p-6 lg:p-8 focus:outline-none"
                tabIndex={-1}
                role="main"
                aria-label="Main content"
              >
                {children}
              </main>
            </div>
          </div>
          
          {/* Live region for screen reader announcements */}
          <div
            aria-live="polite"
            aria-atomic="true"
            className="sr-only"
            id="live-region"
          />
          
          {/* Toast container */}
          <div
            id="toast-container"
            className="fixed top-4 right-4 z-50 space-y-2"
            aria-live="assertive"
            aria-atomic="true"
          />
        </Providers>
      </body>
    </html>
  );
}