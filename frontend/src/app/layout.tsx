import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { ThemeProvider } from "next-themes";
import { Suspense } from "react";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { Topbar } from "@/components/layout/topbar";
import { Toaster } from "@/components/ui/sonner";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "DocuLens | Insurance Document Intelligence",
  description:
    "Automated policy and claim intake using digital image processing, OCR and rule-based extraction.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full bg-muted/30">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <div className="flex min-h-screen">
            <Suspense fallback={<aside className="hidden w-60 shrink-0 border-r bg-background md:block" />}>
              <Sidebar />
            </Suspense>
            <div className="flex min-w-0 flex-1 flex-col">
              <Topbar />
              <main className="mx-auto w-full max-w-7xl flex-1 p-4 md:p-8">{children}</main>
              <footer className="border-t px-8 py-3 text-xs text-muted-foreground">
                This system assists data entry. It does not approve or reject insurance claims.
              </footer>
            </div>
          </div>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
