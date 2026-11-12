export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";
// Static page images are served from the API origin (e.g. /files/12/page_1/thumb.jpg).
export const FILES_URL = API_URL.replace(/\/api\/v1\/?$/, "");

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

function toApiError(status: number, body: unknown): ApiError {
  const err = (body as { error?: { code?: string; message?: string } } | null)?.error;
  return new ApiError(err?.code ?? "HTTP_ERROR", err?.message ?? `Request failed (${status})`, status);
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError("NETWORK", "Cannot reach the server. Is the backend running?", 0);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) throw toApiError(res.status, body);
  return body as T;
}

export interface UploadedItem {
  id: number;
  filename: string;
  status: string;
  page_count: number;
  duplicate_of: number | null;
}
export interface UploadResponse {
  items: UploadedItem[];
  rejected: { filename: string; code: string; message: string }[];
}
export interface StatusOut {
  status: "uploaded" | "processing" | "ingested" | "processed" | "failed";
  stage: string | null;
  progress: number;
}
export interface ClassificationApi {
  category: string;
  confidence: number;
  method: "keyword" | "ml" | "hybrid";
  probabilities: Record<string, number>;
  evidence: { keyword: string; weight: number; count: number; fuzzy: boolean }[];
  negatives?: string[];
  keyword?: { category: string; confidence: number };
  ml?: { category: string; confidence: number };
  note?: string;
  classified_at?: string;
}
export interface FieldApi {
  field_name: string;
  field_value: string | null;
  original_value: string | null;
  confidence: number;
  source_page: number | null;
  bbox_json: { bbox: number[] | null; method: string | null; source: string | null; note: string | null } | null;
  is_edited: boolean;
}
export interface FieldSpecApi {
  name: string;
  label: string;
  kind: "text" | "name" | "date" | "amount" | "id";
  mandatory: boolean;
}
export interface IssueApi {
  rule_code: string;
  field_name: string | null;
  severity: "error" | "warning" | "info";
  message: string;
}
export interface DocumentDetailApi {
  id: number;
  filename: string;
  status: StatusOut["status"];
  page_count: number;
  error_message: string | null;
  doc_category: string | null;
  category_confidence: number | null;
  classification_method: string | null;
  classification_json: ClassificationApi | null;
  review_status: "valid" | "needs_review" | "incomplete" | null;
  issues: IssueApi[];
  fields: FieldApi[];
  pages: {
    page_number: number;
    stage_paths_json: Record<string, string> | null;
    avg_confidence: number | null;
    low_conf_count: number | null;
    ocr_source: string | null;
  }[];
}

/** Uploads one file with byte-level progress (fetch cannot report upload progress, XHR can). */
export function uploadFile(file: File, onProgress: (pct: number) => void): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/documents`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(Math.round((e.loaded / e.total) * 100));
    xhr.onerror = () => reject(new ApiError("NETWORK", "Cannot reach the server. Is the backend running?", 0));
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as UploadResponse);
      else reject(toApiError(xhr.status, body));
    };
    const form = new FormData();
    form.append("files", file);
    xhr.send(form);
  });
}

export interface QualityApi {
  sharpness: number;
  contrast: number;
  noise: number;
  width: number;
  height: number;
  skew_deg?: number;
}
export interface StepLog {
  name: string;
  technique: string;
  opencv: string;
  purpose: string;
  params: Record<string, unknown>;
  ms: number;
  notes: Record<string, unknown>;
}
export interface StageResult {
  page: number;
  recipe: string;
  active_recipe?: string;
  label: string;
  description: string;
  params: Record<string, Record<string, unknown>>;
  total_ms: number;
  stages: { key: string; label: string; url: string; skipped: boolean }[];
  extras?: Record<string, string>;
  steps: StepLog[];
  metrics: { before: QualityApi; after: QualityApi; hist_before: number[]; hist_after: number[] };
  info: Record<string, unknown>;
}
export interface RecipeInfo {
  name: string;
  label: string;
  description: string;
  default: boolean;
  steps: { name: string; technique: string; opencv: string; purpose: string; params: Record<string, unknown> }[];
}

export const getRecipes = () => request<{ default: string; recipes: RecipeInfo[] }>("/pipeline/recipes");
export const getStages = (id: number, page: number, recipe?: string) =>
  request<StageResult>(`/documents/${id}/pages/${page}/stages${recipe ? `?recipe=${recipe}` : ""}`);
export const runPreprocess = (
  id: number,
  recipe: string,
  params: Record<string, Record<string, unknown>> = {},
  pages?: number[],
) =>
  request<{ recipe: string; pages: StageResult[] }>(`/documents/${id}/preprocess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ recipe, params, pages }),
  });

export interface OcrWordApi {
  text: string;
  conf: number; // 0..100
  x: number; // boxes are fractions (0..1) of the OCR'd image
  y: number;
  w: number;
  h: number;
  line: number;
  block: number;
}
export interface OcrPageApi {
  page: number;
  done: boolean;
  source: "ocr" | "text_layer" | null;
  psm: number | null;
  image_url: string | null;
  avg_confidence: number | null;
  word_count: number;
  low_conf_count: number;
  text: string;
  words: OcrWordApi[];
  ocr_ms: number | null;
}
export interface OcrPayload {
  document_id: number;
  threshold: number;
  avg_confidence: number | null;
  low_conf_count: number;
  psm_options: Record<string, string>;
  pages: OcrPageApi[];
}

export const getOcr = (id: number) => request<OcrPayload>(`/documents/${id}/ocr`);
export const runOcr = (
  id: number,
  body: { psm?: number; source?: "processed" | "original"; pages?: number[]; force?: boolean },
) =>
  request<OcrPayload>(`/documents/${id}/ocr`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export const getStatus = (id: number) => request<StatusOut>(`/documents/${id}/status`);
export const getDocument = (id: number) => request<DocumentDetailApi>(`/documents/${id}`);

export const classifyDocument = (id: number, method?: "keyword" | "ml" | "hybrid") =>
  request<ClassificationApi>(`/documents/${id}/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method }),
  });
export const setCategory = (id: number, category: string) =>
  request<DocumentDetailApi>(`/documents/${id}/fields`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ category }),
  });
export const getClassificationInfo = () =>
  request<{ categories: string[]; methods: string[]; default_method: string }>("/classification/info");

export const getExtractionSchema = () => request<Record<string, FieldSpecApi[]>>("/extraction/schema");
export const saveFields = (id: number, fields: Record<string, string | null>) =>
  request<DocumentDetailApi>(`/documents/${id}/fields`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fields }),
  });
export const reExtract = (id: number) =>
  request<{ category: string }>(`/documents/${id}/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });

export const revalidate = (id: number) =>
  request<{ status: string; counts: Record<string, number>; issues: IssueApi[] }>(`/documents/${id}/validate`, { method: "POST" });

// ---- list, stats, delete, report, reprocess ------------------------------------------------
export interface DocumentSummaryApi {
  id: number;
  filename: string;
  file_type: string;
  page_count: number;
  status: StatusOut["status"];
  review_status: "valid" | "needs_review" | "incomplete" | null;
  doc_category: string | null;
  category_confidence: number | null;
  uploaded_at: string;
  processed_at: string | null;
  processing_ms: number | null;
}
export interface DocumentListApi {
  items: DocumentSummaryApi[];
  total: number;
  page: number;
  size: number;
}
export interface ListParams {
  q?: string;
  type?: string;
  status?: string;
  page?: number;
  size?: number;
}
export interface StatsApi {
  total: number;
  valid: number;
  needs_review: number;
  incomplete: number;
  failed: number;
  avg_processing_ms: number | null;
  distribution: { type: string; count: number }[];
  per_day: { day: string; uploads: number; avg_ms: number | null }[];
}

export const listDocuments = (p: ListParams = {}) => {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(p)) if (v !== undefined && v !== "") qs.set(k, String(v));
  return request<DocumentListApi>(`/documents${qs.size ? `?${qs}` : ""}`);
};
export const getStats = () => request<StatsApi>("/stats/summary");
export const deleteDocument = (id: number) => request<null>(`/documents/${id}`, { method: "DELETE" });
export const reprocessDocument = (id: number) => request<{ id: number; status: string }>(`/documents/${id}/process`, { method: "POST" });
export const reportUrl = (id: number, format: "json" | "pdf") => `${API_URL}/documents/${id}/report?format=${format}`;
