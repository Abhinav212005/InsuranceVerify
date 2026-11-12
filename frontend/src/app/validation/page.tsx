"use client";

import Link from "next/link";
import { PartyPopper } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { CategoryBadge, EmptyState, ErrorState, PageHeader, Skeleton, StatusBadge, TimeAgo } from "@/components/common";
import { listDocuments, type DocumentSummaryApi } from "@/lib/api";
import { useAsync } from "@/lib/use-async";
import type { DocCategory } from "@/types/api";

function Queue({ title, status, items }: { title: string; status: "incomplete" | "needs_review"; items: DocumentSummaryApi[] }) {
  return (
    <Card>
      <CardContent className="pt-2">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-medium">{title}</h2>
          <StatusBadge status={status} />
        </div>
        {items.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">Nothing here.</p>
        ) : (
          <ul className="divide-y">
            {items.map((d) => (
              <li key={d.id}>
                <Link
                  href={`/documents/${d.id}?tab=validation`}
                  className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2.5 text-sm hover:bg-muted/50"
                >
                  <span className="min-w-0 flex-1 truncate font-medium">{d.filename}</span>
                  <CategoryBadge category={d.doc_category as DocCategory | null} />
                  <span className="w-20 text-right text-xs text-muted-foreground">
                    <TimeAgo iso={d.uploaded_at} />
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

export default function ValidationCenterPage() {
  const incomplete = useAsync(() => listDocuments({ status: "incomplete", size: 100 }), []);
  const review = useAsync(() => listDocuments({ status: "needs_review", size: 100 }), []);
  const error = incomplete.error ?? review.error;
  const ready = incomplete.data && review.data;
  const empty = ready && incomplete.data!.total === 0 && review.data!.total === 0;

  return (
    <>
      <PageHeader
        title="Validation Center"
        description="Documents waiting for a human check. Open one to correct fields; it is re-validated as you save."
      />
      {error && !ready ? (
        <ErrorState message={error} onRetry={() => { incomplete.reload(); review.reload(); }} />
      ) : !ready ? (
        <Skeleton className="h-48" />
      ) : empty ? (
        <EmptyState icon={PartyPopper} title="Nothing to review" text="Every processed document is complete and consistent." />
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          <Queue title="Incomplete: required fields missing" status="incomplete" items={incomplete.data!.items} />
          <Queue title="Needs review: warnings or low confidence" status="needs_review" items={review.data!.items} />
        </div>
      )}
    </>
  );
}
