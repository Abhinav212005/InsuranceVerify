"use client";

import { useEffect, useMemo, useState } from "react";
import { Copy, Download, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FILES_URL, getOcr, runOcr, type OcrPageApi, type OcrPayload } from "@/lib/api";
import { cn } from "@/lib/utils";

const PSM_LABEL: Record<number, string> = {
  3: "3 · Automatic (default)",
  4: "4 · Single column",
  6: "6 · Uniform block (forms)",
  11: "11 · Sparse text",
  12: "12 · Sparse text + OSD",
};
const selectCls =
  "h-8 rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50";

export function OcrLive({ docId }: { docId: number }) {
  const [data, setData] = useState<OcrPayload | null>(null);
  const [pageIdx, setPageIdx] = useState(0);
  const [threshold, setThreshold] = useState(60);
  const [showAll, setShowAll] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const [psm, setPsm] = useState(3);
  const [source, setSource] = useState<"processed" | "original">("processed");
  const [force, setForce] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getOcr(docId)
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setThreshold(d.threshold);
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : "Request failed"));
    return () => {
      cancelled = true;
    };
  }, [docId]);

  const page: OcrPageApi | undefined = data?.pages[pageIdx];
  const lines = useMemo(() => {
    const out: { idx: number; word: OcrPageApi["words"][number] }[][] = [];
    page?.words.forEach((word, idx) => {
      (out[word.line] ??= []).push({ idx, word });
    });
    return out.filter(Boolean);
  }, [page]);
  const lowCount = page ? page.words.filter((w) => w.conf < threshold).length : 0;

  const rerun = async () => {
    setBusy(true);
    setError(null);
    try {
      setData(await runOcr(docId, { psm, source, force, pages: [pageIdx + 1] }));
      toast.success(`OCR finished for page ${pageIdx + 1}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "OCR failed");
    } finally {
      setBusy(false);
    }
  };

  if (!data)
    return error ? (
      <p role="alert" className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-500/15 dark:text-red-300">
        {error}
      </p>
    ) : (
      <p className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" /> Loading OCR results…
      </p>
    );

  const download = () => {
    const blob = new Blob([data.pages.map((p) => `--- Page ${p.page} ---\n${p.text}`).join("\n\n")], {
      type: "text/plain",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `document_${docId}_ocr.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="flex flex-wrap items-center gap-3 pt-2 text-sm">
          {data.pages.length > 1 && (
            <div className="flex gap-1" role="tablist" aria-label="Pages">
              {data.pages.map((p, i) => (
                <Button key={p.page} size="sm" variant={i === pageIdx ? "default" : "outline"} onClick={() => setPageIdx(i)}>
                  Page {p.page}
                </Button>
              ))}
            </div>
          )}
          <label className="flex items-center gap-2">
            Segmentation (PSM)
            <select className={selectCls} value={psm} onChange={(e) => setPsm(Number(e.target.value))}>
              {Object.entries(PSM_LABEL).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2">
            Read from
            <select className={selectCls} value={source} onChange={(e) => setSource(e.target.value as typeof source)}>
              <option value="processed">processed image</option>
              <option value="original">original image</option>
            </select>
          </label>
          {page?.source === "text_layer" && (
            <label className="flex items-center gap-1.5 text-xs">
              <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
              OCR instead of PDF text
            </label>
          )}
          <Button size="sm" disabled={busy} onClick={() => void rerun()}>
            {busy && <Loader2 className="size-3.5 animate-spin" />} {page?.done ? "Re-run OCR" : "Run OCR"}
          </Button>
          <div className="ml-auto flex gap-1">
            <Button
              size="sm"
              variant="outline"
              disabled={!page?.text}
              onClick={() => {
                void navigator.clipboard.writeText(page?.text ?? "");
                toast.success("Page text copied");
              }}
            >
              <Copy className="size-3.5" /> Copy
            </Button>
            <Button size="sm" variant="outline" onClick={download}>
              <Download className="size-3.5" /> .txt
            </Button>
          </div>
          {error && (
            <p role="alert" className="w-full text-xs text-red-600">
              {error}
            </p>
          )}
        </CardContent>
      </Card>

      {page && !page.done ? (
        <p className="rounded-lg border bg-background p-6 text-center text-sm text-muted-foreground">
          OCR has not been run for this page yet.
        </p>
      ) : (
        page && (
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader className="flex-row items-center justify-between">
                <CardTitle>Image with word boxes</CardTitle>
                <label className="flex items-center gap-1.5 text-xs font-normal">
                  <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} /> show all words
                </label>
              </CardHeader>
              <CardContent>
                <div className="relative mx-auto max-w-md overflow-hidden rounded-lg border bg-white">
                  {page.image_url && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={`${FILES_URL}${page.image_url}`} alt={`Page ${page.page} as read by OCR`} className="block h-auto w-full" />
                  )}
                  <div className="absolute inset-0" aria-hidden>
                    {page.words.map((w, i) => {
                      const low = w.conf < threshold;
                      if (!low && !showAll && hover !== i) return null;
                      return (
                        <div
                          key={i}
                          title={`${w.text} (${w.conf}%)`}
                          className={cn(
                            "absolute border",
                            low
                              ? "border-red-500 bg-red-500/20"
                              : w.conf < threshold + 15
                                ? "border-amber-500 bg-amber-400/10"
                                : "border-emerald-500/60",
                            hover === i && "ring-2 ring-indigo-500",
                          )}
                          style={{ left: `${w.x * 100}%`, top: `${w.y * 100}%`, width: `${w.w * 100}%`, height: `${w.h * 100}%` }}
                        />
                      );
                    })}
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex flex-wrap items-center gap-2">
                  Extracted text
                  <Badge variant="secondary">{page.source === "text_layer" ? "PDF text layer" : `Tesseract · PSM ${page.psm}`}</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
                  <span>
                    Avg confidence: <b className="text-foreground">{page.avg_confidence ?? "–"}%</b>
                  </span>
                  <span>
                    Words: <b className="text-foreground">{page.word_count}</b>
                  </span>
                  <span>
                    Low confidence: <b className="text-foreground">{lowCount}</b>
                  </span>
                  {page.ocr_ms != null && <span>{page.ocr_ms} ms</span>}
                  <label className="ml-auto flex items-center gap-2">
                    Threshold {threshold}
                    <input
                      type="range"
                      min={30}
                      max={95}
                      value={threshold}
                      onChange={(e) => setThreshold(Number(e.target.value))}
                      aria-label="Low confidence threshold"
                      className="accent-indigo-600"
                    />
                  </label>
                </div>
                {page.words.length === 0 ? (
                  <p className="rounded-lg border bg-muted/30 p-4 text-sm text-muted-foreground">
                    No text was found on this page. It may be blank, or the image quality is too low.
                  </p>
                ) : (
                  <div className="max-h-[32rem] overflow-auto rounded-lg border bg-muted/30 p-4 font-mono text-sm leading-7">
                    {lines.map((line, i) => (
                      <div key={i}>
                        {line.map(({ idx, word }) => (
                          <span
                            key={idx}
                            title={`confidence ${word.conf}%`}
                            onMouseEnter={() => setHover(idx)}
                            onMouseLeave={() => setHover(null)}
                            className={cn(
                              "mr-1.5 cursor-default rounded px-0.5",
                              word.conf < threshold &&
                                "bg-red-100 text-red-800 underline decoration-wavy dark:bg-red-500/20 dark:text-red-200",
                              hover === idx && "outline outline-2 outline-indigo-500",
                            )}
                          >
                            {word.text}
                          </span>
                        ))}
                      </div>
                    ))}
                  </div>
                )}
                <p className="text-xs text-muted-foreground">
                  Highlighted words are below the threshold. Hover a word to find it on the image.
                </p>
              </CardContent>
            </Card>
          </div>
        )
      )}
    </div>
  );
}
