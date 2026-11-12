import { Suspense } from "react";
import { DocumentDetail } from "@/components/document-detail";

async function Loader({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <DocumentDetail id={Number(id)} />;
}

export default function Page(props: PageProps<"/documents/[id]">) {
  return (
    <Suspense fallback={<p className="text-sm text-muted-foreground">Loading…</p>}>
      <Loader params={props.params} />
    </Suspense>
  );
}
