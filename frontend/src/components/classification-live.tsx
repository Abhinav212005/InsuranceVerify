"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CategoryBadge } from "@/components/common";
import {
  classifyDocument,
  getClassificationInfo,
  getDocument,
  setCategory,
  type DocumentDetailApi,
} from "@/lib/api";
import { CATEGORY_COLOR, CATEGORY_LABEL } from "@/lib/constants";
import type { DocCategory } from "@/types/api";

const selectCls =
  "h-8 rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50";
const LOW_CONFIDENCE = 0.5;

export function ClassificationLive({
  doc,
  onChange,
}: {
  doc: DocumentDetailApi;
  onChange: (d: DocumentDetailApi) => void;
}) {
  const [methods, setMethods] = useState<string[]>(["keyword"]);
  const [method, setMethod] = useState("");
  const [override, setOverride] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getClassificationInfo()
      .then((i) => setMethods(i.methods))
      .catch(() => {});
  }, []);

  const c = doc.classification_json;
  const category = doc.doc_category as DocCategory | null;
  const confidence = doc.category_confidence ?? 0;
  const manual = doc.classification_method === "manual";

  const act = async (fn: () => Promise<unknown>, ok: string) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChange(await getDocument(doc.id));
      toast.success(ok);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  };

  if (!category)
    return (
      <Card>
        <CardHeader>
          <CardTitle>Classification</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <p className="text-muted-foreground">This document has not been classified yet (it needs OCR text first).</p>
          <Button
            size="sm"
            disabled={busy}
            onClick={() => void act(() => classifyDocument(doc.id), "Document classified")}
          >
            {busy && <Loader2 className="size-3.5 animate-spin" />} Classify now
          </Button>
          {error && (
            <p role="alert" className="text-xs text-red-600">
              {error}
            </p>
          )}
        </CardContent>
      </Card>
    );

  const probs = Object.entries(c?.probabilities ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Document classification</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-base font-medium">
            <CategoryBadge category={category} />
          </span>
          <span className="tabular-nums">
            Confidence: <b>{(confidence * 100).toFixed(0)}%</b>
          </span>
          <Badge variant="secondary" className="capitalize">
            {manual ? "manual override" : doc.classification_method}
          </Badge>
          {!manual && confidence < LOW_CONFIDENCE && (
            <Badge variant="secondary" className="gap-1 bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300">
              <AlertTriangle className="size-3" /> Low confidence: please check
            </Badge>
          )}
        </div>

        {c && !manual && (
          <>
            <div className="space-y-1.5" role="img" aria-label="Probability for each document type">
              {probs.map(([k, p]) => (
                <div key={k} className="flex items-center gap-2 text-xs">
                  <span className="w-32 shrink-0 truncate">{CATEGORY_LABEL[k as DocCategory] ?? k}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${p * 100}%`, background: CATEGORY_COLOR[k as DocCategory] ?? "#64748b" }}
                    />
                  </div>
                  <span className="w-9 text-right tabular-nums text-muted-foreground">{(p * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>

            {c.keyword && c.ml && (
              <p className="text-xs text-muted-foreground">
                Keyword rules say <b>{CATEGORY_LABEL[c.keyword.category as DocCategory]}</b> ({(c.keyword.confidence * 100).toFixed(0)}%),
                the ML model says <b>{CATEGORY_LABEL[c.ml.category as DocCategory]}</b> ({(c.ml.confidence * 100).toFixed(0)}%).
              </p>
            )}

            <div>
              <p className="mb-1.5 text-xs font-medium text-muted-foreground">Evidence found in the text</p>
              {c.evidence.length === 0 ? (
                <p className="text-xs text-muted-foreground">No keywords matched. The document looks unrelated to the known types.</p>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {c.evidence.map((e) => (
                    <span key={e.keyword} className="rounded bg-muted px-2 py-0.5 text-xs" title={`weight ${e.weight}, found ${e.count}×`}>
                      {e.keyword}
                      {e.fuzzy && <i className="text-muted-foreground"> (approx.)</i>}
                    </span>
                  ))}
                </div>
              )}
            </div>
            {c.note && <p className="text-xs text-muted-foreground">{c.note}</p>}
          </>
        )}

        <div className="flex flex-wrap items-end gap-4 border-t pt-4">
          <div className="flex items-center gap-2">
            <select className={selectCls} aria-label="Classification method" value={method} onChange={(e) => setMethod(e.target.value)}>
              <option value="">default method</option>
              {methods.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <Button
              size="sm"
              variant="outline"
              disabled={busy}
              onClick={() =>
                void act(
                  () => classifyDocument(doc.id, (method || undefined) as "keyword" | "ml" | "hybrid" | undefined),
                  "Re-classified",
                )
              }
            >
              {busy && <Loader2 className="size-3.5 animate-spin" />} Re-classify
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <select className={selectCls} aria-label="Override category" value={override} onChange={(e) => setOverride(e.target.value)}>
              <option value="">override category…</option>
              {Object.entries(CATEGORY_LABEL).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
            <Button
              size="sm"
              disabled={busy || !override}
              onClick={() => void act(() => setCategory(doc.id, override), "Category updated")}
            >
              Apply
            </Button>
          </div>
        </div>
        {error && (
          <p role="alert" className="text-xs text-red-600">
            {error}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
