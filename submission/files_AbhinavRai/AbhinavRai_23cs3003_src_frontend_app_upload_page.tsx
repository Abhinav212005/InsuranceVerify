"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { AlertTriangle, CheckCircle2, CircleAlert, FileText, Loader2, UploadCloud, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { PageHeader } from "@/components/common";
import { ApiError, FILES_URL, getDocument, getStatus, uploadFile } from "@/lib/api";
import { PIPELINE_STAGE_LABEL } from "@/lib/constants";
import { cn } from "@/lib/utils";

const ACCEPT = ["application/pdf", "image/png", "image/jpeg"];
const MAX_MB = 20;

type Phase = "uploading" | "processing" | "ready" | "error";
interface Item {
  id: string;
  file: File;
  preview: string | null;
  phase: Phase;
  uploadPct: number;
  stage: string;
  docId?: number;
  pages?: number;
  duplicateOf?: number | null;
  error?: string;
}

const STAGE_LABEL = PIPELINE_STAGE_LABEL;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function barValue(it: Item) {
  if (it.phase === "uploading") return it.uploadPct * 0.5;
  if (it.phase === "processing") {
    const at: Record<string, number> = { uploaded: 55, rendering: 65, rendered: 70, preprocessing: 78, preprocessed: 85, ocr: 90, ocr_done: 93, classifying: 95, classified: 96, extracting: 97, extracted: 98, validating: 99 };
    return at[it.stage] ?? 60;
  }
  return it.phase === "ready" ? 100 : 0;
}

export default function UploadPage() {
  const [items, setItems] = useState<Item[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const patch = useCallback((id: string, p: Partial<Item>) => {
    if (alive.current) setItems((prev) => prev.map((i) => (i.id === id ? { ...i, ...p } : i)));
  }, []);

  const run = useCallback(
    async (item: Item) => {
      try {
        const res = await uploadFile(item.file, (pct) => patch(item.id, { uploadPct: pct }));
        const created = res.items[0];
        if (!created) throw new ApiError("REJECTED", res.rejected[0]?.message ?? "File rejected", 400);
        patch(item.id, {
          phase: "processing",
          docId: created.id,
          pages: created.page_count,
          duplicateOf: created.duplicate_of,
          stage: "uploaded",
        });
        for (let i = 0; i < 150 && alive.current; i++) {
          const s = await getStatus(created.id);
          patch(item.id, { stage: s.stage ?? "uploaded" });
          if (s.status === "failed") {
            const d = await getDocument(created.id);
            throw new ApiError("PROCESSING_FAILED", d.error_message ?? "Processing failed", 500);
          }
          if (s.status === "ingested" || s.status === "processed") {
            const d = await getDocument(created.id);
            const thumb = d.pages[0]?.stage_paths_json?.thumb;
            const done: Partial<Item> = { phase: "ready", stage: s.stage ?? "ocr_done", pages: d.page_count };
            // PDFs have no browser preview, so use the server-rendered first page
            if (item.file.type === "application/pdf" && thumb) done.preview = `${FILES_URL}${thumb}`;
            patch(item.id, done);
            return;
          }
          await sleep(800);
        }
        throw new ApiError("TIMEOUT", "Processing is taking too long", 0);
      } catch (e) {
        patch(item.id, { phase: "error", error: e instanceof Error ? e.message : "Upload failed" });
      }
    },
    [patch],
  );

  const addFiles = useCallback(
    (list: FileList | File[]) => {
      const next: Item[] = [];
      const errs: string[] = [];
      for (const file of Array.from(list)) {
        if (!ACCEPT.includes(file.type)) errs.push(`${file.name}: unsupported type (PDF, PNG, JPG only)`);
        else if (file.size > MAX_MB * 1024 * 1024) errs.push(`${file.name}: larger than ${MAX_MB} MB`);
        else
          next.push({
            id: `${file.name}-${file.size}-${Math.random()}`,
            file,
            preview: file.type.startsWith("image/") ? URL.createObjectURL(file) : null,
            phase: "uploading",
            uploadPct: 0,
            stage: "uploaded",
          });
      }
      setError(errs.length ? errs.join(" · ") : null);
      setItems((p) => [...p, ...next]);
      next.forEach((it) => void run(it));
    },
    [run],
  );

  return (
    <>
      <PageHeader
        title="Upload documents"
        description="PDF, PNG, JPG or JPEG up to 20 MB each (PDFs up to 10 pages). Multiple files supported."
      />
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          addFiles(e.dataTransfer.files);
        }}
        className={cn(
          "flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed bg-background p-12 text-center transition-colors",
          over ? "border-indigo-500 bg-indigo-50/50 dark:bg-indigo-500/10" : "border-border",
        )}
      >
        <span className="flex size-12 items-center justify-center rounded-full bg-indigo-50 text-indigo-600 dark:bg-indigo-500/15">
          <UploadCloud className="size-6" />
        </span>
        <p className="font-medium">Drag and drop files here</p>
        <p className="text-sm text-muted-foreground">or</p>
        <Button onClick={() => inputRef.current?.click()}>Browse files</Button>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.png,.jpg,.jpeg"
          className="hidden"
          onChange={(e) => {
            if (e.target.files) addFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {error && (
        <p role="alert" className="mt-3 rounded-lg bg-red-50 px-4 py-2 text-sm text-red-700 dark:bg-red-500/15 dark:text-red-300">
          {error}
        </p>
      )}

      <div className="mt-6 space-y-3">
        {items.map((it) => (
          <Card key={it.id}>
            <CardContent className="flex items-center gap-4 pt-2">
              {it.preview ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={it.preview} alt={`Preview of ${it.file.name}`} className="size-14 rounded-lg border object-cover" />
              ) : (
                <span className="flex size-14 items-center justify-center rounded-lg border bg-muted">
                  <FileText className="size-6 text-muted-foreground" />
                </span>
              )}
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <p className="truncate text-sm font-medium">{it.file.name}</p>
                  <span
                    className={cn(
                      "flex shrink-0 items-center gap-1 text-xs",
                      it.phase === "error" ? "text-red-600" : "text-muted-foreground",
                    )}
                  >
                    {it.phase === "ready" && <CheckCircle2 className="size-3.5 text-emerald-600" />}
                    {it.phase === "error" && <CircleAlert className="size-3.5" />}
                    {(it.phase === "uploading" || it.phase === "processing") && <Loader2 className="size-3.5 animate-spin" />}
                    {it.phase === "uploading"
                      ? `Uploading ${it.uploadPct}%`
                      : it.phase === "error"
                        ? "Failed"
                        : (STAGE_LABEL[it.stage] ?? it.stage)}
                  </span>
                </div>
                <Progress value={barValue(it)} className="mt-2" />
                <p className="mt-1 text-xs text-muted-foreground">
                  {(it.file.size / 1024).toFixed(0)} KB
                  {it.pages ? ` · ${it.pages} page${it.pages > 1 ? "s" : ""}` : ""}
                </p>
                {it.phase === "error" && (
                  <p role="alert" className="mt-1 text-xs text-red-600">
                    {it.error}
                  </p>
                )}
                {it.phase === "ready" && it.docId && (
                  <Link
                    href={`/documents/${it.docId}`}
                    className="mt-1 inline-block text-xs font-medium text-indigo-600 hover:underline dark:text-indigo-300"
                  >
                    Open document →
                  </Link>
                )}
                {it.duplicateOf && (
                  <p className="mt-1 flex items-center gap-1 text-xs text-amber-700 dark:text-amber-300">
                    <AlertTriangle className="size-3.5" /> Identical to an earlier upload (document #{it.duplicateOf}).
                  </p>
                )}
              </div>
              <Button
                variant="ghost"
                size="icon"
                aria-label={`Remove ${it.file.name} from list`}
                onClick={() => setItems((p) => p.filter((x) => x.id !== it.id))}
              >
                <X className="size-4" />
              </Button>
            </CardContent>
          </Card>
        ))}
        {items.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Pages are rendered at up to 300 DPI, cleaned with OpenCV and read with Tesseract OCR and classified by document type. Fields are then extracted and validated automatically.
          </p>
        )}
      </div>
    </>
  );
}
