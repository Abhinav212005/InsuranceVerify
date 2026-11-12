"use client";

import { ErrorState } from "@/components/common";

export default function Error({ error, reset }: { error: Error; reset: () => void }) {
  return <ErrorState message={error.message || "Something went wrong while showing this page."} onRetry={reset} />;
}
