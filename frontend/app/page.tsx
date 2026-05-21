"use client";

import { useState } from "react";

import { PdfPane } from "@/components/PdfPane";
import { SummaryPane } from "@/components/SummaryPane";
import { UploadPanel } from "@/components/UploadPanel";
import type {
  ActiveSource,
  CaseManifest,
  OnePageSummary,
} from "@/lib/types";

export default function Home() {
  const [manifest, setManifest] = useState<CaseManifest | null>(null);
  const [summary, setSummary] = useState<OnePageSummary | null>(null);
  const [active, setActive] = useState<ActiveSource>(null);

  if (!manifest || !summary) {
    return (
      <UploadPanel
        onReady={(m, s) => {
          setManifest(m);
          setSummary(s);
        }}
      />
    );
  }

  return (
    <main className="h-screen flex">
      <div className="w-1/2 overflow-auto p-4 border-r border-clinical-200">
        <SummaryPane summary={summary} active={active} onPick={setActive} />
      </div>
      <div className="w-1/2 h-full">
        <PdfPane caseId={manifest.case_id} manifest={manifest} active={active} />
      </div>
    </main>
  );
}
