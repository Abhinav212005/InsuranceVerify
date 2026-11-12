"use client";

import Link from "next/link";
import { useSyncExternalStore } from "react";
import { AlertTriangle, CheckCircle2, CircleAlert, Loader2, RefreshCw, type LucideIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { CATEGORY_COLOR, CATEGORY_LABEL, STATUS_LABEL } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { DocCategory, ReviewStatus } from "@/types/api";

const STATUS_STYLE: Record<ReviewStatus, string> = {
  valid: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300",
  needs_review: "bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300",
  incomplete: "bg-red-50 text-red-700 dark:bg-red-500/15 dark:text-red-300",
};
const STATUS_ICON = { valid: CheckCircle2, needs_review: AlertTriangle, incomplete: CircleAlert };

export function StatusBadge({ status }: { status: ReviewStatus }) {
  const Icon = STATUS_ICON[status];
  return (
    <Badge variant="secondary" className={cn("gap-1 border-0", STATUS_STYLE[status])}>
      <Icon className="size-3" />
      {STATUS_LABEL[status]}
    </Badge>
  );
}

/** Badge for a document row: processing and failure states take precedence over the review status. */
export function DocStatusBadge({ status, review }: { status: string; review: ReviewStatus | null }) {
  if (status === "failed")
    return (
      <Badge variant="secondary" className="gap-1 border-0 bg-red-50 text-red-700 dark:bg-red-500/15 dark:text-red-300">
        <CircleAlert className="size-3" /> Failed
      </Badge>
    );
  if (status === "uploaded" || status === "processing")
    return (
      <Badge variant="secondary" className="gap-1 border-0">
        <Loader2 className="size-3 animate-spin" /> Processing
      </Badge>
    );
  return review ? <StatusBadge status={review} /> : <Badge variant="secondary">Not validated</Badge>;
}

export function CategoryBadge({ category }: { category: DocCategory | null }) {
  if (!category) return <span className="text-sm text-muted-foreground">Unclassified</span>;
  return (
    <span className="inline-flex items-center gap-2 text-sm">
      <span className="size-2 rounded-full" style={{ background: CATEGORY_COLOR[category] }} />
      {CATEGORY_LABEL[category]}
    </span>
  );
}

export function PageHeader({ title, description, children }: { title: string; description?: string; children?: React.ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-start justify-between gap-3">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 text-sm text-muted-foreground">{description}</p>}
      </div>
      {children}
    </div>
  );
}

const noopSubscribe = () => () => {};

export function TimeAgo({ iso }: { iso: string }) {
  // The API sends UTC without a zone suffix; treat it as UTC. The server render shows the date and the
  // browser swaps in the relative time (keeps prerendering deterministic).
  const utc = iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`;
  const text = useSyncExternalStore(
    noopSubscribe,
    () => {
      const min = Math.floor((Date.now() - new Date(utc).getTime()) / 60000);
      if (min < 1) return "just now";
      if (min < 60) return `${min} min ago`;
      const h = Math.floor(min / 60);
      return h < 24 ? `${h}h ago` : `${Math.floor(h / 24)}d ago`;
    },
    () => iso.slice(0, 10),
  );
  return <span suppressHydrationWarning>{text}</span>;
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-center gap-3 rounded-xl border border-red-200 bg-red-50 p-8 text-center text-sm text-red-800 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-200">
      <CircleAlert className="size-6" />
      <p className="max-w-md">{message}</p>
      {onRetry && (
        <Button size="sm" variant="outline" onClick={onRetry}>
          <RefreshCw className="size-3.5" /> Try again
        </Button>
      )}
    </div>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  text,
  href,
  action,
}: {
  icon: LucideIcon;
  title: string;
  text?: string;
  href?: string;
  action?: string;
}) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed bg-background p-10 text-center">
      <span className="flex size-11 items-center justify-center rounded-full bg-muted text-muted-foreground">
        <Icon className="size-5" />
      </span>
      <p className="font-medium">{title}</p>
      {text && <p className="max-w-sm text-sm text-muted-foreground">{text}</p>}
      {href && action && (
        <Link href={href} className={cn(buttonVariants({ size: "sm" }), "mt-2")}>
          {action}
        </Link>
      )}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-md bg-muted", className)} />;
}
