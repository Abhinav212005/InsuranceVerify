"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, Download, FileQuestion, Loader2, RotateCw, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ClassificationLive } from "@/components/classification-live";
import { EmptyState, ErrorState, PageHeader, Skeleton, StatusBadge } from "@/components/common";
import { FieldsLive } from "@/components/fields-live";
import { LiveOverview } from "@/components/live-overview";
import { OcrLive } from "@/components/ocr-live";
import { StudioLive } from "@/components/studio-live";
import { ValidationLive } from "@/components/validation-live";
import { deleteDocument, getDocument, getStatus, reportUrl, reprocessDocument, type StatusOut } from "@/lib/api";
import { PIPELINE_STAGE_LABEL } from "@/lib/constants";
import { useAsync } from "@/lib/use-async";
import { cn } from "@/lib/utils";

export function DocumentDetail({ id }: { id: number }) {
  const router = useRouter();
  const [tab, setTab] = useState(useSearchParams().get("tab") ?? "overview");
  const doc = useAsync(() => getDocument(id), [id]);
  const [progress, setProgress] = useState<StatusOut | null>(null);
  const [busy, setBusy] = useState(false);

  const status = doc.data?.status;
  const inFlight = status === "uploaded" || status === "processing";
  const reload = doc.reload;

  // while the pipeline runs, follow its progress and load the result when it finishes
  useEffect(() => {
    if (!inFlight) return;
    const t = setInterval(() => {
      getStatus(id)
        .then((s) => {
          setProgress(s);
          if (s.status !== "uploaded" && s.status !== "processing") reload();
        })
        .catch(() => {});
    }, 1500);
    return () => clearInterval(t);
  }, [inFlight, id, reload]);

  const act = async (fn: () => Promise<unknown>, ok: string) => {
    setBusy(true);
    try {
      await fn();
      toast.success(ok);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  };

  if (doc.error && !doc.data)
    return doc.error.toLowerCase().includes("not found") ? (
      <EmptyState icon={FileQuestion} title="Document not found" text="It may have been deleted." href="/documents" action="Back to documents" />
    ) : (
      <ErrorState message={doc.error} onRetry={doc.reload} />
    );
  if (!doc.data) return <Skeleton className="h-64" />;

  const d = doc.data;
  const reprocess = () =>
    act(async () => {
      await reprocessDocument(d.id);
      setProgress(null);
      reload();
    }, "Processing started again");

  return (
    <>
      <Link href="/documents" className="mb-3 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="size-4" /> Documents
      </Link>
      <PageHeader title={d.filename} description={`${d.page_count} page${d.page_count > 1 ? "s" : ""}`}>
        <div className="flex flex-wrap items-center gap-2">
          {d.review_status && !inFlight && <StatusBadge status={d.review_status} />}
          {!inFlight && d.status !== "failed" && (
            <>
              <a href={reportUrl(d.id, "pdf")} download className={cn(buttonVariants({ variant: "outline", size: "sm" }))}>
                <Download className="size-4" /> PDF report
              </a>
              <a href={reportUrl(d.id, "json")} download className={cn(buttonVariants({ variant: "outline", size: "sm" }))}>
                <Download className="size-4" /> JSON
              </a>
              <Button
                size="sm"
                variant="outline"
                disabled={busy}
                onClick={() => window.confirm("Run the whole pipeline again? Your corrections are kept.") && void reprocess()}
              >
                <RotateCw className="size-4" /> Re-process
              </Button>
            </>
          )}
          <Button
            size="sm"
            variant="outline"
            disabled={busy || inFlight}
            onClick={() => {
              if (!window.confirm(`Delete “${d.filename}”? This cannot be undone.`)) return;
              void act(async () => {
                await deleteDocument(d.id);
                router.push("/documents");
              }, "Document deleted");
            }}
          >
            <Trash2 className="size-4" /> Delete
          </Button>
        </div>
      </PageHeader>

      {inFlight ? (
        <Card>
          <CardContent className="space-y-3 pt-2">
            <p className="flex items-center gap-2 text-sm font-medium">
              <Loader2 className="size-4 animate-spin" /> {PIPELINE_STAGE_LABEL[progress?.stage ?? "uploaded"] ?? "Processing"}…
            </p>
            <Progress value={progress?.progress ?? 5} />
            <p className="text-xs text-muted-foreground">This page updates by itself when processing finishes.</p>
          </CardContent>
        </Card>
      ) : d.status === "failed" ? (
        <div className="space-y-3">
          <ErrorState message={d.error_message ?? "Processing failed."} />
          <Button disabled={busy} onClick={() => void reprocess()}>
            <RotateCw className="size-4" /> Try processing again
          </Button>
        </div>
      ) : (
        <>
          {d.error_message && (
            <p role="alert" className="mb-4 rounded-lg bg-amber-50 px-4 py-2 text-sm text-amber-800 dark:bg-amber-500/15 dark:text-amber-200">
              {d.error_message} Use “Re-process” to try again.
            </p>
          )}
          <Tabs value={tab} onValueChange={(v) => setTab(String(v))}>
            <TabsList className="mb-4 flex-wrap">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="studio">Image Studio</TabsTrigger>
              <TabsTrigger value="ocr">OCR</TabsTrigger>
              <TabsTrigger value="intelligence">Intelligence</TabsTrigger>
              <TabsTrigger value="validation">Validation</TabsTrigger>
            </TabsList>
            <TabsContent value="overview">
              <LiveOverview doc={d} />
            </TabsContent>
            <TabsContent value="studio">
              <StudioLive docId={d.id} pageCount={d.page_count} />
            </TabsContent>
            <TabsContent value="ocr">
              <OcrLive docId={d.id} />
            </TabsContent>
            <TabsContent value="intelligence">
              <div className="space-y-4">
                <ClassificationLive doc={d} onChange={() => reload()} />
                <FieldsLive doc={d} onChange={() => reload()} />
              </div>
            </TabsContent>
            <TabsContent value="validation">
              <ValidationLive doc={d} onChange={() => reload()} />
            </TabsContent>
          </Tabs>
        </>
      )}
    </>
  );
}
