"use client";

import { useTheme } from "next-themes";
import { useRouter } from "next/navigation";
import { Moon, Search, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export function Topbar() {
  const { resolvedTheme, setTheme } = useTheme();
  const router = useRouter();
  return (
    <header className="sticky top-0 z-10 flex h-14 items-center gap-3 border-b bg-background/80 px-4 backdrop-blur md:px-8">
      <form
        className="relative w-full max-w-md"
        onSubmit={(e) => {
          e.preventDefault();
          const q = new FormData(e.currentTarget).get("q")?.toString().trim();
          router.push(q ? `/documents?q=${encodeURIComponent(q)}` : "/documents");
        }}
      >
        <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          name="q"
          aria-label="Search documents"
          placeholder="Search by filename, policy no., name…"
          className="pl-9"
        />
      </form>
      <div className="ml-auto">
        <Button
          variant="ghost"
          size="icon"
          aria-label="Toggle theme"
          onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
        >
          <Sun className="size-4 dark:hidden" />
          <Moon className="hidden size-4 dark:block" />
        </Button>
      </div>
    </header>
  );
}
