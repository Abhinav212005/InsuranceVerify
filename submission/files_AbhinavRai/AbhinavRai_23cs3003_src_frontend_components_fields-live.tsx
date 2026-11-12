"use client";

import { useEffect, useMemo, useState } from "react";
import { Crosshair, Eye, EyeOff, Loader2, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  FILES_URL,
  getDocument,
  getExtractionSchema,
  reExtract,
  saveFields,
  type DocumentDetailApi,
  type FieldApi,
  type FieldSpecApi,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const LOW = 0.6;
const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

function ConfBar({ value }: { value: number }) {
  const color = value >= 0.8 ? "bg-emerald-500" : value >= LOW ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2" title={`Confidence ${(value * 100).toFixed(0)}%`}>
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full", color)} style={{ width: `${value * 100}%` }} />
      </div>
      <span className="w-9 text-xs tabular-nums text-muted-foreground">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

const mask = (v: string) => (v.length > 4 ? `${"•".repeat(v.length - 4)}${v.slice(-4)}` : v);

export function FieldsLive({ doc, onChange }: { doc: DocumentDetailApi; onChange: (d: DocumentDetailApi) => void }) {
  const [specs, setSpecs] = useState<FieldSpecApi[]>([]);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [focus, setFocus] = useState<string | null>(null);
  const [reveal, setReveal] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getExtractionSchema()
      .then((s) => setSpecs(s[doc.doc_category ?? "other"] ?? []))
      .catch(() => {});
  }, [doc.doc_category]);

  const rows = useMemo(() => {
    const byName = new Map(doc.fields.map((f) => [f.field_name, f]));
    return specs.map((s) => ({ spec: s, field: byName.get(s.name) as FieldApi | undefined }));
  }, [specs, doc.fields]);

  const current = (name: string, f?: FieldApi) => draft[name] ?? f?.field_value ?? "";
  const dirty = rows.filter(({ spec, field }) => draft[spec.name] !== undefined && draft[spec.name] !== (field?.field_value ?? ""));
  const focusRow = rows.find((r) => r.spec.name === focus)?.field;
  const box = focusRow?.bbox_json?.bbox ?? null;
  const page = doc.pages.find((p) => p.page_number === (focusRow?.source_page ?? 1)) ?? doc.pages[0];
  const imageUrl = page?.stage_paths_json?.ocr_image ?? page?.stage_paths_json?.original;
  const sensitive = doc.doc_category === "identity_document";

  const run = async (fn: () => Promise<unknown>, ok: string) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChange(await getDocument(doc.id));
      setDraft({});
      toast.success(ok);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  };

  if (!doc.doc_category || specs.length === 0)
    return (
      <Card>
        <CardHeader>
          <CardTitle>Extracted fields</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          {doc.doc_category === "other"
            ? "Documents of type “Other Supporting” have no structured fields. Change the category above if this is wrong."
            : "Classify the document first; the fields to extract depend on its type."}
        </CardContent>
      </Card>
    );

  return (
    <div className="grid gap-4 lg:grid-cols-5">
      <Card className="lg:col-span-3">
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle>Extracted fields</CardTitle>
          <div className="flex gap-2">
            {sensitive && (
              <Button size="sm" variant="outline" onClick={() => setReveal((r) => !r)}>
                {reveal ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />} {reveal ? "Hide" : "Show"} ID
              </Button>
            )}
            <Button
              size="sm"
              variant="outline"
              disabled={busy}
              onClick={() => void run(() => reExtract(doc.id), "Fields re-extracted (your edits were kept)")}
            >
              <RefreshCw className="size-3.5" /> Re-extract
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {rows.map(({ spec, field }) => {
            const missing = !field?.field_value;
            const low = !missing && !field!.is_edited && field!.confidence < LOW;
            const note = field?.bbox_json?.note;
            const hide = sensitive && spec.kind === "id" && !reveal;
            const value = current(spec.name, field);
            return (
              <div key={spec.name} className="grid items-start gap-x-3 gap-y-1 sm:grid-cols-[170px_1fr]">
                <label htmlFor={spec.name} className="pt-2 text-sm font-medium">
                  {spec.label}
                  {spec.mandatory && <span className="text-red-500" title="required"> *</span>}
                </label>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <Input
                      id={spec.name}
                      type={spec.kind === "date" && ISO_DATE.test(value) ? "date" : "text"}
                      value={hide ? mask(value) : value}
                      readOnly={hide}
                      placeholder={missing ? "Not found: enter manually" : ""}
                      aria-invalid={missing && spec.mandatory ? true : undefined}
                      onFocus={() => setFocus(spec.name)}
                      onChange={(e) => setDraft((d) => ({ ...d, [spec.name]: e.target.value }))}
                    />
                    <Button
                      size="icon"
                      variant="ghost"
                      aria-label={`Show ${spec.label} on the image`}
                      disabled={!field?.bbox_json?.bbox}
                      onClick={() => setFocus(spec.name)}
                    >
                      <Crosshair className="size-4" />
                    </Button>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    {field?.is_edited ? (
                      <Badge variant="secondary">{field.field_value === field.original_value ? "confirmed" : "edited"}</Badge>
                    ) : missing ? (
                      <Badge variant="secondary" className="text-amber-700 dark:text-amber-300">
                        not found
                      </Badge>
                    ) : (
                      <ConfBar value={field!.confidence} />
                    )}
                    {low && <span className="text-amber-700 dark:text-amber-300">low confidence: please check</span>}
                    {note && <span className="text-muted-foreground">{note}</span>}
                    {draft[spec.name] !== undefined && draft[spec.name] !== (field?.field_value ?? "") && (
                      <span className="text-indigo-600">unsaved</span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
          {error && (
            <p role="alert" className="text-xs text-red-600">
              {error}
            </p>
          )}
          <div className="flex gap-2 border-t pt-4">
            <Button
              disabled={busy || dirty.length === 0}
              onClick={() =>
                void run(
                  () => saveFields(doc.id, Object.fromEntries(dirty.map(({ spec }) => [spec.name, draft[spec.name] || null]))),
                  `Saved ${dirty.length} correction${dirty.length > 1 ? "s" : ""}`,
                )
              }
            >
              {busy && <Loader2 className="size-3.5 animate-spin" />} Save corrections
            </Button>
            <Button variant="outline" disabled={Object.keys(draft).length === 0} onClick={() => setDraft({})}>
              Reset
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Source on page {page?.page_number ?? 1}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative overflow-hidden rounded-lg border bg-white">
            {imageUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={`${FILES_URL}${imageUrl}`} alt={`Page ${page?.page_number ?? 1} of the document`} className="block h-auto w-full" />
            ) : (
              <div className="aspect-[3/4]" />
            )}
            {box && (
              <div
                aria-hidden
                className="absolute border-2 border-indigo-500 bg-indigo-500/20"
                style={{ left: `${box[0] * 100}%`, top: `${box[1] * 100}%`, width: `${box[2] * 100}%`, height: `${box[3] * 100}%` }}
              />
            )}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            {box
              ? `Highlighted: where “${rows.find((r) => r.spec.name === focus)?.spec.label}” was read.`
              : "Click a field (or its target icon) to see where its value was found."}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
