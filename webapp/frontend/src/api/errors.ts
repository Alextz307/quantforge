function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export interface ValidationItem {
  loc: readonly unknown[];
  msg: string;
}

export function isValidationItem(value: unknown): value is ValidationItem {
  return isObject(value) && Array.isArray(value.loc) && typeof value.msg === "string";
}

export function extractValidationItems(error: unknown): readonly ValidationItem[] {
  if (!isObject(error)) return [];
  const detail = error.detail;
  if (!Array.isArray(detail)) return [];
  return detail.filter(isValidationItem);
}

function formatField(loc: readonly unknown[]): string {
  const tail = loc[loc.length - 1];
  if (typeof tail === "string" || typeof tail === "number") return String(tail);
  return "field";
}

export function extractApiError(error: unknown, fallback: string): string {
  if (!isObject(error)) return fallback;
  const detail = error.detail;
  if (typeof detail === "string") return detail;
  const items = extractValidationItems(error);
  if (items.length > 0) {
    return items.map((item) => `${formatField(item.loc)}: ${item.msg}`).join("; ");
  }
  return fallback;
}

export interface LocatedError {
  loc: readonly (string | number)[];
  msg: string;
}

export function buildErrorIndex(errors: readonly LocatedError[]): ReadonlyMap<string, string> {
  const out = new Map<string, string>();
  for (const err of errors) out.set(err.loc.join("."), err.msg);
  return out;
}
