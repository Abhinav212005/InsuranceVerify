"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CategoryBadge } from "@/components/common";
import { FILES_URL, type DocumentDetailApi } from "@/lib/api";
import type { DocCategory } from "@/types/api";

export function LiveOverview({ doc }: { doc: DocumentDetailApi }) {
  const page = doc.pages[0];
  const paths = page?.stage_paths_json ?? {};
  const confs = doc.pages.map((p) => p.avg_confidence).filter((c): c is number => c != null);
  const avg = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : null;
  const low = doc.pages.reduce((n, p) => n + (p.low_conf_count ?? 0), 0);
  const c = doc.classification_json;

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card>
        <CardHeader>
          <CardTitle>Original → Processed (page 1)</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-2">
          {(
            [
              ["original", "Original"],
              ["final", "Processed"],
            ] as const
          ).map(([k, label]) => (
            <figure key={k}>
              <div className="aspect-[3/4] overflow-hidden rounded-lg border bg-white">
                {paths[k] && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={`${FILES_URL}${paths[k]}`} alt={`${label} page 1`} className="h-full w-full object-contain" />
                )}
              </div>
              <figcaption className="mt-1 text-center text-xs text-muted-foreground">{label}</figcaption>
            </figure>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Classification</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          {doc.doc_category ? (
            <>
              <CategoryBadge category={doc.doc_category as DocCategory} />
              <p>
                Confidence: <b>{((doc.category_confidence ?? 0) * 100).toFixed(0)}%</b>
                <span className="ml-2 text-xs capitalize text-muted-foreground">({doc.classification_method})</span>
              </p>
              {c && c.evidence.length > 0 && (
                <div>
                  <p className="mb-1 text-xs text-muted-foreground">Matched keywords</p>
                  <div className="flex flex-wrap gap-1">
                    {c.evidence.slice(0, 6).map((e) => (
                      <span key={e.keyword} className="rounded bg-muted px-2 py-0.5 text-xs">
                        {e.keyword}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <p className="text-muted-foreground">Not classified yet.</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Text recognition</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <p>
            Average OCR confidence: <b>{avg != null ? `${avg.toFixed(1)}%` : "not run"}</b>
          </p>
          <p>
            Low-confidence words: <b>{low}</b>
          </p>
          <p>
            Fields found: <b>{doc.fields.filter((f) => f.field_value).length}</b> of {doc.fields.length}
          </p>
          <p>
            Issues: <b>{doc.issues.filter((i) => i.severity !== "info").length}</b>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
