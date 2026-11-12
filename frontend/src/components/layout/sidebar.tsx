"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  UploadCloud,
  FileStack,
  ShieldCheck,
  ScanSearch,
  FlaskConical,
} from "lucide-react";
import { getStats } from "@/lib/api";
import { useAsync } from "@/lib/use-async";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/upload", label: "Upload", icon: UploadCloud },
  { href: "/documents", label: "Documents", icon: FileStack },
  { href: "/validation", label: "Validation Center", icon: ShieldCheck },
  { href: "/evaluation", label: "Evaluation", icon: FlaskConical },
];

export function Sidebar() {
  const pathname = usePathname();
  const stats = useAsync(getStats, [pathname]);
  const queue = stats.data ? stats.data.needs_review + stats.data.incomplete : 0;
  return (
    <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r bg-background md:flex">
      <div className="flex h-14 items-center gap-2 border-b px-5">
        <span className="flex size-8 items-center justify-center rounded-lg bg-indigo-600 text-white">
          <ScanSearch className="size-4" />
        </span>
        <span className="font-semibold tracking-tight">DocuLens</span>
      </div>
      <nav className="flex-1 space-y-1 p-3">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-indigo-50 text-indigo-700 dark:bg-indigo-500/15 dark:text-indigo-300"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )}
            >
              <Icon className="size-4" />
              {label}
              {href === "/validation" && queue > 0 && (
                <span className="ml-auto rounded-full bg-amber-100 px-1.5 text-xs tabular-nums text-amber-800 dark:bg-amber-500/20 dark:text-amber-200" title="Documents waiting for review">
                  {queue}
                </span>
              )}
            </Link>
          );
        })}
      </nav>
      <div className="border-t p-4 text-xs text-muted-foreground">
        Insurance Document Intelligence
        <br />
        DIP project · v1.0
      </div>
    </aside>
  );
}
