"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, CircleAlert, Info, Loader2, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { StatusBadge } from "@/components/common";
import {
  getDocument,
  getExtractionSchema,
  revalidate,
  saveFields,
  type DocumentDetailApi,
  type FieldSpecApi,
  type IssueApi,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const MISSING = new Set(["MISSING_FIELD", "POLICY_NUMBER_MISSING", "CLAIM_NUMBER_MISSING"]);

const SUMMARY = {
  valid: "All required data is present and no problems were found.",
  needs_review: "Required data is present, but some values look wrong or uncertain. Please check them.",
  incomplete: "Required fields are missing. Fill them in below or in the Intelligence tab.",
} as const;

function IssueRow({
  issue,
  label,
  onFix,
  busy,
}: {
  issue: IssueApi;
  label: string | null;
  onFix: (field: string, value: string) => void;
  busy: boolean;
}) {
  const [value, setValue] = useState("");
  const Icon = issue.severity === "error" ? CircleAlert : issue.severity === "warning" ? AlertTriangle : Info;
  const tone = issue.severity === "error" ? "text-red-600" : issue.severity === "warning" ? "text-amber-600" : "text-sky-600";
  const fixable = issue.field_name && issue.severity !== "info";
  return (
    <li className="rounded-lg border bg-background p-3 text-sm">
      <div className="flex items-start gap-3">
        <Icon className={cn("mt-0.5 size-4 shrink-0", tone)} />
        <div className="min-w-0 flex-1">
          <p>{issue.message}</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {issue.rule_code}
            {label && ` · ${label}`}
          </p>
        </div>
      </div>
      {fixable && (
        <form
          className="mt-2 flex gap-2 pl-7"
          onSubmit={(e) => {
            e.preventDefault();
            if (value.trim()) onFix(issue.field_name!, value.trim());
          }}
        >
          <Input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            aria-label={`Corrected value for ${label ?? issue.field_name}`}
            placeholder={`Enter the correct ${label?.toLowerCase() ?? "value"}`}
            className="h-8 max-w-xs"
          />
          <Button size="sm" type="submit" disabled={busy || !value.trim()}>
            Fix
          </Button>
        </form>
      )}
    </li>
  );
}

export function ValidationLive({ doc, onChange }: { doc: DocumentDetailApi; onChange: (d: DocumentDetailApi) => void }) {
  const [specs, setSpecs] = useState<FieldSpecApi[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getExtractionSchema()
      .then((s) => setSpecs(s[doc.doc_category ?? "other"] ?? []))
      .catch(() => {});
  }, [doc.doc_category]);

  const labelOf = (f: string | null) => (f ? (specs.find((s) => s.name === f)?.label ?? f) : null);
  const run = async (fn: () => Promise<unknown>, ok: string) => {
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

  const status = doc.review_status;
  const missing = doc.issues.filter((i) => MISSING.has(i.rule_code));
  const attention = doc.issues.filter((i) => !MISSING.has(i.rule_code) && i.severity !== "info");
  const info = doc.issues.filter((i) => i.severity === "info");

  if (!status)
    return (
      <Card>
        <CardHeader>
          <CardTitle>Validation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <p>This document has not been validated yet (it needs OCR text or a category first).</p>
          <Button size="sm" disabled={busy} onClick={() => void run(() => revalidate(doc.id), "Validated")}>
            {busy && <Loader2 className="size-3.5 animate-spin" />} Validate now
          </Button>
          {error && <p role="alert" className="text-xs text-red-600">{error}</p>}
        </CardContent>
      </Card>
    );

  const section = (title: string, items: IssueApi[]) =>
    items.length > 0 && (
      <div>
        <h3 className="mb-2 flex items-center gap-2 text-sm font-medium">
          {title} <Badge variant="secondary">{items.length}</Badge>
        </h3>
        <ul className="space-y-2">
          {items.map((i, k) => (
            <IssueRow
              key={`${i.rule_code}-${i.field_name}-${k}`}
              issue={i}
              label={labelOf(i.field_name)}
              busy={busy}
              onFix={(field, value) => void run(() => saveFields(doc.id, { [field]: value }), `${labelOf(field)} updated and re-checked`)}
            />
          ))}
        </ul>
      </div>
    );

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="flex flex-wrap items-center gap-4 pt-2">
          <div className="flex items-center gap-3">
            <StatusBadge status={status} />
            <p className="text-sm">{SUMMARY[status]}</p>
          </div>
          <Button size="sm" variant="outline" className="ml-auto" disabled={busy} onClick={() => void run(() => revalidate(doc.id), "Re-validated")}>
            {busy ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />} Re-validate
          </Button>
        </CardContent>
      </Card>

      {status === "valid" && doc.issues.every((i) => i.severity === "info") && (
        <p className="flex items-center gap-2 rounded-lg border bg-background p-4 text-sm">
          <CheckCircle2 className="size-4 text-emerald-600" /> No problems found.
        </p>
      )}
      {section("Missing required fields", missing)}
      {section("Needs attention", attention)}
      {section("For your information", info)}
      {error && <p role="alert" className="text-xs text-red-600">{error}</p>}
      <p className="text-xs text-muted-foreground">
        “Valid” means the data is complete and consistent. It is not an approval, and this system never approves or rejects insurance claims.
      </p>
    </div>
  );
}
