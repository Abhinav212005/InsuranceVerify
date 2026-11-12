"use client";

import { FileQuestion } from "lucide-react";
import { EmptyState } from "@/components/common";

export default function NotFound() {
  return <EmptyState icon={FileQuestion} title="Page not found" text="The page you opened does not exist." href="/" action="Go to the dashboard" />;
}
