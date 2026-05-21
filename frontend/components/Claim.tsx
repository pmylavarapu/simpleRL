"use client";

import type { Claim, ActiveSource } from "@/lib/types";

interface Props {
  claim: Claim;
  active: ActiveSource;
  onPick: (src: ActiveSource) => void;
  inline?: boolean;
}

function isActive(claim: Claim, active: ActiveSource): boolean {
  if (!active) return false;
  return claim.sources.some(
    (s) =>
      s.pdf_id === active.pdf_id &&
      s.page === active.page &&
      s.snippet === active.snippet
  );
}

export function ClaimSpan({ claim, active, onPick, inline = false }: Props) {
  const handleClick = () => {
    if (claim.sources.length > 0) onPick(claim.sources[0]);
  };
  const cls = `claim${isActive(claim, active) ? " active" : ""}`;
  const Tag = inline ? "span" : "div";
  return (
    <Tag
      className={cls}
      onClick={handleClick}
      title={
        claim.sources[0]
          ? `Source: page ${claim.sources[0].page + 1} — “${claim.sources[0].snippet.slice(
              0,
              80
            )}”`
          : undefined
      }
    >
      {claim.text}
    </Tag>
  );
}
