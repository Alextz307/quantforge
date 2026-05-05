export function formatDateTime(iso: string): string {
  return new Date(iso).toISOString().replace("T", " ").slice(0, 19) + " UTC";
}

export function formatDate(iso: string): string {
  return iso.slice(0, 10);
}

export function formatMetric(value: number | null | undefined, digits = 4): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

export function formatPercent(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function shortHash(hash: string, prefixLen = 12): string {
  return hash.length > prefixLen ? `${hash.slice(0, prefixLen)}…` : hash;
}

export function withCi(mean: number, low: number, high: number, digits = 3): string {
  return `${formatMetric(mean, digits)} [${formatMetric(low, digits)}, ${formatMetric(high, digits)}]`;
}
