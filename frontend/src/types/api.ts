export type DocCategory =
  | "insurance_policy"
  | "claim_form"
  | "identity_document"
  | "medical_bill"
  | "repair_invoice"
  | "other";

export type ReviewStatus = "valid" | "needs_review" | "incomplete";

export interface StageInfo {
  key: "original" | "gray" | "denoised" | "thresholded" | "deskewed" | "final";
  label: string;
  technique: string;
  opencv: string;
  explanation: string;
}
