"use client";

import { FlaskConical } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState, ErrorState, PageHeader, Skeleton } from "@/components/common";
import { API_URL, request } from "@/lib/api";
import { useAsync } from "@/lib/use-async";

interface Summary {
  ablation?: { mean_cer: Record<string, number>; n_images: number };
  skew?: { mean_angle_error_deg: number };
  photo?: { boundary_found_rate: number; mean_iou: number };
  classification?: { overall_accuracy: Record<string, number>; n: number };
  fields?: { overall_oracle: number; overall_pipeline: number };
  validation?: { status_accuracy: number; cases: number };
  timing?: Record<string, { total_ms: number }>;
}
interface Payload {
  summary: Summary;
  charts: string[];
}

const TITLES: Record<string, string> = {
  "01_ablation_bar": "1. Preprocessing ablation",
  "01_ablation_heatmap": "1. Ablation by degradation",
  "02_threshold": "2. Thresholding methods",
  "03_denoise": "3. Denoising methods",
  "04_skew": "4. Skew correction",
  "05_resolution": "5. Scan resolution",
  "06_photo": "6. Phone photos",
  "07_psm": "7. Page segmentation mode",
  "08_classification": "8. Classification by degradation",
  "08_confusion": "8. Confusion matrix",
  "08_classifier_heldout": "8. Classifier on an unseen layout",
  "09_fields_heatmap": "9. Field accuracy by type and degradation",
  "09_fields_bar": "9. Field accuracy: true vs predicted category",
  "10_timing": "10. Processing time",
  "11_validation": "11. Validation engine",
};
const pct = (x?: number) => (x == null ? "–" : `${(x * 100).toFixed(1)}%`);

export default function EvaluationPage() {
  const data = useAsync(() => request<Payload>("/evaluation/summary"), []);

  if (data.error && !data.data)
    return (
      <>
        <PageHeader title="Evaluation" />
        {data.error.includes("No evaluation") ? (
          <EmptyState icon={FlaskConical} title="No evaluation results yet" text="Run scripts/evaluation/run_experiments.py and make_charts.py, then reload this page." />
        ) : (
          <ErrorState message={data.error} onRetry={data.reload} />
        )}
      </>
    );
  if (!data.data) return <Skeleton className="h-64" />;

  const { summary: s, charts } = data.data;
  const mc = s.ablation?.mean_cer ?? {};
  const raw = mc["A0 raw image"];
  const best = Math.min(...Object.values(mc).filter((x) => !Number.isNaN(x)));
  const timing = Object.entries(s.timing ?? {});
  const kpis = [
    { label: "OCR error: raw → best pipeline", value: raw != null ? `${raw.toFixed(3)} → ${best.toFixed(3)}` : "–", hint: "mean CER, all degradations" },
    { label: "Skew angle error", value: s.skew ? `${s.skew.mean_angle_error_deg.toFixed(2)}°` : "–", hint: "mean absolute error" },
    { label: "Page found in photos", value: pct(s.photo?.boundary_found_rate), hint: s.photo ? `mean IoU ${s.photo.mean_iou.toFixed(2)}` : "" },
    { label: "Classification (hybrid)", value: pct(s.classification?.overall_accuracy?.hybrid), hint: `${s.classification?.n ?? 0} degraded images` },
    { label: "Field extraction", value: pct(s.fields?.overall_oracle), hint: `with predicted type: ${pct(s.fields?.overall_pipeline)}` },
    { label: "Validation status accuracy", value: pct(s.validation?.status_accuracy), hint: `${s.validation?.cases ?? 0} injected-defect cases` },
  ];

  return (
    <>
      <PageHeader
        title="Evaluation"
        description="Results of the experiments in scripts/evaluation/ on a synthetic benchmark (no real personal data). Numbers measure this system on this benchmark, not real-world accuracy."
      />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {kpis.map((k) => (
          <Card key={k.label}>
            <CardContent className="pt-2">
              <p className="text-sm text-muted-foreground">{k.label}</p>
              <p className="mt-1 text-2xl font-semibold tabular-nums">{k.value}</p>
              <p className="text-xs text-muted-foreground">{k.hint}</p>
            </CardContent>
          </Card>
        ))}
      </div>
      {timing.length > 0 && (
        <p className="mt-3 text-sm text-muted-foreground">
          Average time per page:{" "}
          {timing.map(([res, t], i) => (
            <span key={res}>
              {i > 0 && " · "}
              <b>{(t.total_ms / 1000).toFixed(1)} s</b> at {res}
            </span>
          ))}
        </p>
      )}
      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        {charts.map((c) => {
          const key = c.replace(".png", "");
          return (
            <Card key={c} className={key.includes("heatmap") || key.includes("ablation_bar") ? "lg:col-span-2" : undefined}>
              <CardHeader>
                <CardTitle>{TITLES[key] ?? key}</CardTitle>
              </CardHeader>
              <CardContent>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`${API_URL}/evaluation/charts/${c}`} alt={TITLES[key] ?? key} className="mx-auto h-auto max-w-full rounded-md border bg-white" />
              </CardContent>
            </Card>
          );
        })}
      </div>
    </>
  );
}
