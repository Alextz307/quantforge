function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

interface ValidationItem {
  loc: readonly unknown[];
  msg: string;
}

function isValidationItem(value: unknown): value is ValidationItem {
  return isObject(value) && Array.isArray(value.loc) && typeof value.msg === "string";
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
  if (Array.isArray(detail)) {
    const messages = detail
      .filter(isValidationItem)
      .map((item) => `${formatField(item.loc)}: ${item.msg}`);
    if (messages.length > 0) return messages.join("; ");
  }
  return fallback;
}
