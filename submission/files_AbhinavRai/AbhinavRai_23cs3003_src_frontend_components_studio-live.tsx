"use client";

import { useCallback, useEffect, useState } from "react";
import { ArrowRight, Loader2 } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  FILES_URL,
  getRecipes,
  getStages,
  runPreprocess,
  type QualityApi,
  type RecipeInfo,
  type StageResult,
} from "@/lib/api";
import { STAGES } from "@/lib/constants";
import { cn } from "@/lib/utils";

const STAGE_STEPS: Record<string, string[]> = {
  gray: ["gray"],
  denoised: ["denoise", "contrast"],
  thresholded: ["threshold"],
  deskewed: ["deskew"],
  final: ["morphology", "resize"],
};

const selectCls =
  "h-8 rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50";

const src = (url: string) => `${FILES_URL}${url}`;

function Img({ url, alt, className }: { url: string; alt: string; className?: string }) {
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src(url)} alt={alt} className={cn("h-full w-full bg-white object-contain", className)} />;
}

function MetricRows({ before, after }: { before: QualityApi; after: QualityApi }) {
  const rows = [
    { label: "Sharpness (Laplacian var.)", b: before.sharpness, a: after.sharpness, up: true, unit: "" },
    { label: "Contrast (std dev)", b: before.contrast, a: after.contrast, up: true, unit: "" },
    { label: "Noise estimate (σ)", b: before.noise, a: after.noise, up: false, unit: "" },
    { label: "Skew angle", b: before.skew_deg ?? 0, a: after.skew_deg ?? 0, up: false, unit: "°", abs: true },
  ];
  return (
    <div className="space-y-2.5">
      {rows.map((r) => {
        const bv = r.abs ? Math.abs(r.b) : r.b;
        const av = r.abs ? Math.abs(r.a) : r.a;
        const same = Math.abs(av - bv) < 0.005;
        const better = r.up ? av > bv : av < bv;
        return (
          <div key={r.label} className="flex items-center justify-between gap-2 text-sm">
            <span className="text-muted-foreground">{r.label}</span>
            <span className="flex items-center gap-2 tabular-nums">
              {r.b}
              {r.unit} → <b>{r.a}{r.unit}</b>
              <Badge variant="secondary" className={same ? "" : better ? "text-emerald-700" : "text-amber-700"}>
                {same ? "same" : better ? "better" : "worse"}
              </Badge>
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function StudioLive({ docId, pageCount }: { docId: number; pageCount: number }) {
  const [page, setPage] = useState(1);
  const [recipes, setRecipes] = useState<RecipeInfo[]>([]);
  const [res, setRes] = useState<StageResult | null>(null);
  const [cmp, setCmp] = useState<StageResult | null>(null);
  const [idx, setIdx] = useState(5);
  const [pos, setPos] = useState(50);
  const [mode, setMode] = useState<"slider" | "side">("slider");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [thr, setThr] = useState("");
  const [den, setDen] = useState("");

  const guard = useCallback(async <T,>(fn: () => Promise<T>): Promise<T | null> => {
    setBusy(true);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
      return null;
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void getRecipes().then((r) => setRecipes(r.recipes)).catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    getStages(docId, page)
      .then((r) => {
        if (cancelled) return;
        setRes(r);
        setCmp(null);
        setError(null);
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : "Request failed"));
    return () => {
      cancelled = true;
    };
  }, [docId, page]);

  const run = async (recipe: string, useOverrides: boolean) => {
    const info = recipes.find((r) => r.name === recipe);
    const has = (s: string) => info?.steps.some((x) => x.name === s);
    const params: Record<string, Record<string, unknown>> = {};
    if (useOverrides && thr && has("threshold")) params.threshold = { method: thr };
    if (useOverrides && den && has("denoise")) params.denoise = { method: den };
    return guard(() => runPreprocess(docId, recipe, params, [page]));
  };

  const rerun = async (recipe: string, useOverrides = true) => {
    const r = await run(recipe, useOverrides);
    if (r) {
      setRes(r.pages[0]);
      setCmp(null);
    }
  };

  const compareWith = async (recipe: string) => {
    if (!recipe) return setCmp(null);
    const existing = await guard(() => getStages(docId, page, recipe).catch(() => null));
    if (existing) return setCmp(existing);
    // not run yet: running it makes it the page's active recipe, so restore the current one after
    const keep = res?.recipe;
    const r = await run(recipe, false);
    if (r) setCmp(r.pages[0]);
    if (keep) await guard(() => runPreprocess(docId, keep, res?.params ?? {}, [page]).then((x) => setRes(x.pages[0])));
  };

  if (!res)
    return error ? (
      <p role="alert" className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-500/15 dark:text-red-300">
        {error}
      </p>
    ) : (
      <p className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" /> Loading image processing results…
      </p>
    );

  const stage = res.stages[idx];
  const stageInfo = STAGES[idx];
  const steps = res.steps.filter((s) => STAGE_STEPS[stage.key]?.includes(s.name));
  const hist = res.metrics.hist_before.map((b, i) => ({
    bin: i * 8,
    Before: b,
    After: res.metrics.hist_after[i],
  }));

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="flex flex-wrap items-center gap-3 pt-2 text-sm">
          {pageCount > 1 && (
            <label className="flex items-center gap-2">
              Page
              <select className={selectCls} value={page} onChange={(e) => setPage(Number(e.target.value))}>
                {Array.from({ length: pageCount }, (_, i) => (
                  <option key={i} value={i + 1}>
                    {i + 1}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className="flex items-center gap-2">
            Recipe
            <select
              className={selectCls}
              value={res.recipe}
              disabled={busy}
              onChange={(e) => void rerun(e.target.value, false)}
              aria-label="Preprocessing recipe"
            >
              {recipes.map((r) => (
                <option key={r.name} value={r.name}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2">
            Threshold
            <select className={selectCls} value={thr} onChange={(e) => setThr(e.target.value)}>
              <option value="">recipe default</option>
              <option value="adaptive">adaptive</option>
              <option value="otsu">Otsu</option>
              <option value="global">global</option>
            </select>
          </label>
          <label className="flex items-center gap-2">
            Denoise
            <select className={selectCls} value={den} onChange={(e) => setDen(e.target.value)}>
              <option value="">recipe default</option>
              <option value="median">median</option>
              <option value="gaussian">gaussian</option>
              <option value="nlm">non-local means</option>
              <option value="none">none</option>
            </select>
          </label>
          <Button size="sm" disabled={busy} onClick={() => void rerun(res.recipe)}>
            {busy && <Loader2 className="size-3.5 animate-spin" />} Apply
          </Button>
          <label className="ml-auto flex items-center gap-2">
            Compare with
            <select className={selectCls} value={cmp?.recipe ?? ""} disabled={busy} onChange={(e) => void compareWith(e.target.value)}>
              <option value="">none</option>
              {recipes
                .filter((r) => r.name !== res.recipe)
                .map((r) => (
                  <option key={r.name} value={r.name}>
                    {r.label}
                  </option>
                ))}
            </select>
          </label>
          {error && (
            <p role="alert" className="w-full text-xs text-red-600">
              {error}
            </p>
          )}
          <p className="w-full text-xs text-muted-foreground">
            {res.description} Processed in {res.total_ms} ms.
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-2">
          <Card>
            <CardContent className="pt-2">
              <ol className="flex flex-wrap items-center gap-1" aria-label="Processing stages">
                {res.stages.map((s, i) => (
                  <li key={s.key} className="flex items-center gap-1">
                    <button
                      onClick={() => setIdx(i)}
                      aria-current={i === idx ? "step" : undefined}
                      className={cn(
                        "rounded-lg px-3 py-1.5 text-sm transition-colors",
                        i === idx ? "bg-indigo-600 text-white" : "bg-muted text-muted-foreground hover:text-foreground",
                        s.skipped && i !== idx && "opacity-50",
                      )}
                      title={s.skipped ? "Not part of this recipe" : undefined}
                    >
                      {s.label}
                    </button>
                    {i < res.stages.length - 1 && <ArrowRight className="size-3.5 text-muted-foreground" />}
                  </li>
                ))}
              </ol>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex-row items-center justify-between">
              <CardTitle>Original vs {stage.label}</CardTitle>
              <div className="flex gap-1">
                {(["slider", "side"] as const).map((m) => (
                  <Button key={m} size="sm" variant={mode === m ? "default" : "outline"} onClick={() => setMode(m)}>
                    {m === "slider" ? "Slider" : "Side by side"}
                  </Button>
                ))}
              </div>
            </CardHeader>
            <CardContent>
              {mode === "slider" ? (
                <div>
                  <div className="relative mx-auto aspect-[3/4] max-w-md overflow-hidden rounded-lg border bg-white">
                    <Img url={stage.url} alt={`${stage.label} stage`} className="absolute inset-0" />
                    <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
                      <Img url={res.stages[0].url} alt="Original page" />
                    </div>
                    <div className="absolute inset-y-0 w-0.5 bg-indigo-500 shadow" style={{ left: `${pos}%` }} />
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={pos}
                    onChange={(e) => setPos(Number(e.target.value))}
                    aria-label="Before and after comparison position"
                    className="mx-auto mt-3 block w-full max-w-md accent-indigo-600"
                  />
                  <p className="mt-1 text-center text-xs text-muted-foreground">
                    Left: original · Right: {stage.label}
                    {stage.skipped && " (this stage is not part of the recipe, shows the previous result)"}
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-3">
                  {[res.stages[0], stage].map((s, i) => (
                    <figure key={i}>
                      <div className="aspect-[3/4] overflow-hidden rounded-lg border">
                        <Img url={s.url} alt={s.label} />
                      </div>
                      <figcaption className="mt-1 text-center text-xs text-muted-foreground">{s.label}</figcaption>
                    </figure>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {cmp && (
            <Card>
              <CardHeader>
                <CardTitle>
                  Recipe comparison: {res.label} vs {cmp.label}
                </CardTitle>
              </CardHeader>
              <CardContent className="grid gap-4 sm:grid-cols-2">
                {[res, cmp].map((r) => (
                  <div key={r.recipe} className="space-y-2">
                    <div className="aspect-[3/4] overflow-hidden rounded-lg border">
                      <Img url={r.stages[5].url} alt={`Final image, ${r.label}`} />
                    </div>
                    <p className="text-sm font-medium">{r.label}</p>
                    <MetricRows before={r.metrics.before} after={r.metrics.after} />
                    <p className="text-xs text-muted-foreground">{r.total_ms} ms</p>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {res.extras && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Thresholding comparison</CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-3 gap-3">
                  {(
                    [
                      ["cmp_global", "Global (T = 127)"],
                      ["cmp_otsu", "Otsu"],
                      ["cmp_adaptive", "Adaptive"],
                    ] as const
                  ).map(([k, label]) => (
                    <figure key={k}>
                      <div className="aspect-[3/4] overflow-hidden rounded-lg border">
                        <Img url={res.extras![k]} alt={label} />
                      </div>
                      <figcaption className="mt-1 text-center text-xs text-muted-foreground">{label}</figcaption>
                    </figure>
                  ))}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Morphological operations (on the Otsu image)</CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {(["erode", "dilate", "open", "close"] as const).map((op) => (
                    <figure key={op}>
                      <div className="aspect-[3/4] overflow-hidden rounded-lg border">
                        <Img url={res.extras![`morph_${op}`]} alt={`${op} result`} />
                      </div>
                      <figcaption className="mt-1 text-center text-xs capitalize text-muted-foreground">
                        {op === "open" ? "opening" : op === "close" ? "closing" : op}
                      </figcaption>
                    </figure>
                  ))}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Edges, contours and document boundary</CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-3 gap-3">
                  {(
                    [
                      ["edges", "Canny edges"],
                      ["contours", "Largest contours"],
                      ["boundary", res.info.boundary_detected ? "Page boundary found" : "No page boundary (flat scan)"],
                    ] as const
                  ).map(([k, label]) => (
                    <figure key={k}>
                      <div className="aspect-[3/4] overflow-hidden rounded-lg border">
                        <Img url={res.extras![k]} alt={label} />
                      </div>
                      <figcaption className="mt-1 text-center text-xs text-muted-foreground">{label}</figcaption>
                    </figure>
                  ))}
                </CardContent>
              </Card>
            </>
          )}
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>{stageInfo.technique}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <p>{stageInfo.explanation}</p>
              {stage.key === "original" ? null : steps.length === 0 ? (
                <p className="text-xs text-muted-foreground">This step is not part of the selected recipe.</p>
              ) : (
                steps.map((s) => (
                  <div key={s.name} className="rounded-lg border p-2.5">
                    <div className="flex items-center justify-between text-xs">
                      <b>{s.technique}</b>
                      <span className="text-muted-foreground">{s.ms} ms</span>
                    </div>
                    <code className="mt-1.5 block rounded bg-muted px-2 py-1 text-xs">{s.opencv}</code>
                    {Object.keys(s.params).length > 0 && (
                      <p className="mt-1.5 text-xs text-muted-foreground">
                        {Object.entries(s.params)
                          .map(([k, v]) => `${k}=${String(v)}`)
                          .join(", ")}
                      </p>
                    )}
                    {Object.entries(s.notes).map(([k, v]) => (
                      <p key={k} className="mt-1 text-xs">
                        <span className="text-muted-foreground">{k.replaceAll("_", " ")}:</span> <b>{String(v)}</b>
                      </p>
                    ))}
                  </div>
                ))
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Quality: before → after</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <MetricRows before={res.metrics.before} after={res.metrics.after} />
              <p className="text-xs text-muted-foreground">
                Before = original page, after = processed grayscale (skew measured on the final image).
                Measured at a fixed width so values are comparable. Sharpness can fall after denoising because
                noise itself creates false edges.
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Intensity histogram</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-44" role="img" aria-label="Histogram before and after processing">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={hist}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="bin" tickLine={false} axisLine={false} fontSize={10} />
                    <YAxis hide scale="sqrt" />
                    <Tooltip />
                    <Legend iconSize={8} wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="Before" fill="#94a3b8" />
                    <Bar dataKey="After" fill="#4f46e5" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
