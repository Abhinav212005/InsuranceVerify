"use client";

import Link from "next/link";
import { AlertTriangle, CheckCircle2, CircleAlert, FileText, Inbox } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CategoryBadge, DocStatusBadge, EmptyState, ErrorState, PageHeader, Skeleton, TimeAgo } from "@/components/common";
import { getStats, listDocuments } from "@/lib/api";
import { CATEGORY_COLOR, CATEGORY_LABEL } from "@/lib/constants";
import { useAsync } from "@/lib/use-async";
import type { DocCategory, ReviewStatus } from "@/types/api";

const weekday = (iso: string) => new Date(`${iso}T12:00:00Z`).toLocaleDateString(undefined, { weekday: "short", timeZone: "UTC" });

export default function DashboardPage() {
  const stats = useAsync(getStats, []);
  const recent = useAsync(() => listDocuments({ size: 6 }), []);

  if (stats.error && !stats.data) return (
    <>
      <PageHeader title="Dashboard" />
      <ErrorState message={stats.error} onRetry={stats.reload} />
    </>
  );

  const s = stats.data;
  const kpis = s
    ? [
        { label: "Documents uploaded", value: s.total, icon: FileText, tone: "text-indigo-600 bg-indigo-50 dark:bg-indigo-500/15" },
        { label: "Valid (data complete)", value: s.valid, icon: CheckCircle2, tone: "text-emerald-600 bg-emerald-50 dark:bg-emerald-500/15" },
        { label: "Needs review", value: s.needs_review, icon: AlertTriangle, tone: "text-amber-600 bg-amber-50 dark:bg-amber-500/15" },
        { label: "Incomplete", value: s.incomplete, icon: CircleAlert, tone: "text-red-600 bg-red-50 dark:bg-red-500/15" },
      ]
    : [];
  const dist = (s?.distribution ?? []).map((d) => ({
    ...d,
    name: CATEGORY_LABEL[d.type as DocCategory] ?? d.type,
    fill: CATEGORY_COLOR[d.type as DocCategory] ?? "#64748b",
  }));
  const perDay = (s?.per_day ?? []).map((d) => ({ ...d, label: weekday(d.day) }));

  return (
    <>
      <PageHeader title="Dashboard" description="Overview of document intake and processing." />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {!s
          ? Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-24" />)
          : kpis.map(({ label, value, icon: Icon, tone }) => (
              <Card key={label}>
                <CardContent className="flex items-center justify-between pt-2">
                  <div>
                    <p className="text-sm text-muted-foreground">{label}</p>
                    <p className="mt-1 text-3xl font-semibold tabular-nums">{value}</p>
                  </div>
                  <span className={`flex size-10 items-center justify-center rounded-xl ${tone}`}>
                    <Icon className="size-5" />
                  </span>
                </CardContent>
              </Card>
            ))}
      </div>
      {s && s.failed > 0 && (
        <p className="mt-3 text-sm text-red-700 dark:text-red-300">
          {s.failed} document{s.failed > 1 ? "s" : ""} failed to process.{" "}
          <Link href="/documents" className="underline">
            Open the list
          </Link>{" "}
          to retry.
        </p>
      )}

      {s && s.total === 0 ? (
        <div className="mt-4">
          <EmptyState icon={Inbox} title="No documents yet" text="Upload a policy, claim form, bill or ID to see it processed here." href="/upload" action="Upload documents" />
        </div>
      ) : (
        <>
          <div className="mt-4 grid gap-4 lg:grid-cols-5">
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Document type distribution</CardTitle>
              </CardHeader>
              <CardContent>
                {!s ? (
                  <Skeleton className="h-60" />
                ) : dist.length === 0 ? (
                  <p className="py-16 text-center text-sm text-muted-foreground">Documents appear here once they are classified.</p>
                ) : (
                  <>
                    <div className="h-60" role="img" aria-label="Donut chart of document types">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie data={dist} dataKey="count" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={2}>
                            {dist.map((d) => (
                              <Cell key={d.type} fill={d.fill} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <ul className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                      {dist.map((d) => (
                        <li key={d.type} className="flex items-center gap-2">
                          <span className="size-2 rounded-full" style={{ background: d.fill }} />
                          <span className="truncate">{d.name}</span>
                          <span className="ml-auto tabular-nums text-muted-foreground">{d.count}</span>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </CardContent>
            </Card>

            <Card className="lg:col-span-3">
              <CardHeader>
                <CardTitle>Processing statistics · uploads, last 7 days</CardTitle>
              </CardHeader>
              <CardContent>
                {!s ? (
                  <Skeleton className="h-64" />
                ) : (
                  <>
                    <div className="h-64" role="img" aria-label="Bar chart of uploads per day">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={perDay}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} />
                          <XAxis dataKey="label" tickLine={false} axisLine={false} />
                          <YAxis tickLine={false} axisLine={false} width={30} allowDecimals={false} />
                          <Tooltip cursor={{ fill: "rgba(99,102,241,0.08)" }} />
                          <Bar dataKey="uploads" name="Uploads" fill="#4f46e5" radius={[6, 6, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <p className="mt-2 text-xs text-muted-foreground">
                      Average processing time:{" "}
                      <b>{s.avg_processing_ms != null ? `${(s.avg_processing_ms / 1000).toFixed(1)} s` : "–"}</b> per document.
                    </p>
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Recent activity</CardTitle>
            </CardHeader>
            <CardContent className="divide-y">
              {recent.error && !recent.data ? (
                <ErrorState message={recent.error} onRetry={recent.reload} />
              ) : !recent.data ? (
                <Skeleton className="h-32" />
              ) : (
                recent.data.items.map((d) => (
                  <Link key={d.id} href={`/documents/${d.id}`} className="flex flex-wrap items-center gap-3 py-3 text-sm hover:bg-muted/50">
                    <FileText className="size-4 text-muted-foreground" />
                    <span className="min-w-0 flex-1 truncate font-medium">{d.filename}</span>
                    <CategoryBadge category={d.doc_category as DocCategory | null} />
                    <DocStatusBadge status={d.status} review={d.review_status as ReviewStatus | null} />
                    <span className="w-20 text-right text-xs text-muted-foreground">
                      <TimeAgo iso={d.uploaded_at} />
                    </span>
                  </Link>
                ))
              )}
            </CardContent>
          </Card>
        </>
      )}
    </>
  );
}
