"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { ChevronLeft, ChevronRight, FileSearch, Inbox, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { CategoryBadge, DocStatusBadge, EmptyState, ErrorState, PageHeader, Skeleton, TimeAgo } from "@/components/common";
import { deleteDocument, listDocuments } from "@/lib/api";
import { CATEGORY_LABEL, STATUS_LABEL } from "@/lib/constants";
import { useAsync } from "@/lib/use-async";
import type { DocCategory, ReviewStatus } from "@/types/api";

const SIZE = 10;
const selectCls =
  "h-8 rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50";

function useDebounced<T>(value: T, ms = 300) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

function History() {
  const [q, setQ] = useState(useSearchParams().get("q") ?? "");
  const [type, setType] = useState("");
  const [status, setStatus] = useState("");
  const [page, setPage] = useState(1);
  const dq = useDebounced(q);
  const list = useAsync(() => listDocuments({ q: dq, type, status, page, size: SIZE }), [dq, type, status, page]);

  // keep the table fresh while anything is still being processed
  const active = list.data?.items.some((d) => d.status === "uploaded" || d.status === "processing");
  const reload = list.reload;
  useEffect(() => {
    if (!active) return;
    const t = setInterval(reload, 2000);
    return () => clearInterval(t);
  }, [active, reload]);

  const filtered = Boolean(dq || type || status);
  const totalPages = Math.max(1, Math.ceil((list.data?.total ?? 0) / SIZE));
  const change = (fn: () => void) => {
    fn();
    setPage(1);
  };

  const remove = async (id: number, name: string) => {
    if (!window.confirm(`Delete “${name}”? The file, extracted data and report cannot be recovered.`)) return;
    try {
      await deleteDocument(id);
      toast.success("Document deleted");
      list.reload();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Could not delete the document");
    }
  };

  return (
    <>
      <PageHeader title="Documents" description="Every uploaded document with search and filters.">
        <Link href="/upload" className="text-sm font-medium text-indigo-600 hover:underline dark:text-indigo-300">
          Upload more →
        </Link>
      </PageHeader>

      <div className="mb-4 flex flex-wrap gap-3">
        <Input
          value={q}
          onChange={(e) => change(() => setQ(e.target.value))}
          aria-label="Search documents"
          placeholder="Search filename or extracted values…"
          className="max-w-xs"
        />
        <select aria-label="Document type" className={selectCls} value={type} onChange={(e) => change(() => setType(e.target.value))}>
          <option value="">All types</option>
          {Object.entries(CATEGORY_LABEL).map(([k, v]) => (
            <option key={k} value={k}>
              {v}
            </option>
          ))}
        </select>
        <select aria-label="Status" className={selectCls} value={status} onChange={(e) => change(() => setStatus(e.target.value))}>
          <option value="">All statuses</option>
          {Object.entries(STATUS_LABEL).map(([k, v]) => (
            <option key={k} value={k}>
              {v}
            </option>
          ))}
        </select>
        {filtered && (
          <Button variant="ghost" size="sm" onClick={() => { setQ(""); setType(""); setStatus(""); setPage(1); }}>
            Clear filters
          </Button>
        )}
      </div>

      {list.error && !list.data ? (
        <ErrorState message={list.error} onRetry={list.reload} />
      ) : !list.data ? (
        <Skeleton className="h-64" />
      ) : list.data.total === 0 ? (
        filtered ? (
          <EmptyState icon={FileSearch} title="No documents match your filters" text="Try a different search term or clear the filters." />
        ) : (
          <EmptyState icon={Inbox} title="No documents yet" text="Upload your first document to get started." href="/upload" action="Upload documents" />
        )
      ) : (
        <>
          <Card className="overflow-hidden p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Document</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Pages</TableHead>
                  <TableHead>Uploaded</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="w-10" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {list.data.items.map((d) => (
                  <TableRow key={d.id}>
                    <TableCell>
                      <Link href={`/documents/${d.id}`} className="font-medium hover:underline">
                        {d.filename}
                      </Link>
                    </TableCell>
                    <TableCell>
                      <CategoryBadge category={d.doc_category as DocCategory | null} />
                    </TableCell>
                    <TableCell>{d.page_count}</TableCell>
                    <TableCell className="text-muted-foreground">
                      <TimeAgo iso={d.uploaded_at} />
                    </TableCell>
                    <TableCell>
                      <DocStatusBadge status={d.status} review={d.review_status as ReviewStatus | null} />
                    </TableCell>
                    <TableCell>
                      <Button variant="ghost" size="icon" aria-label={`Delete ${d.filename}`} onClick={() => void remove(d.id, d.filename)}>
                        <Trash2 className="size-4 text-muted-foreground" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
          <div className="mt-3 flex items-center justify-between text-sm text-muted-foreground">
            <span>
              {list.data.total} document{list.data.total > 1 ? "s" : ""}
            </span>
            <div className="flex items-center gap-2">
              <Button size="icon" variant="outline" aria-label="Previous page" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                <ChevronLeft className="size-4" />
              </Button>
              <span className="tabular-nums">
                Page {page} of {totalPages}
              </span>
              <Button size="icon" variant="outline" aria-label="Next page" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
                <ChevronRight className="size-4" />
              </Button>
            </div>
          </div>
        </>
      )}
    </>
  );
}

export default function DocumentsPage() {
  return (
    <Suspense>
      <History />
    </Suspense>
  );
}
