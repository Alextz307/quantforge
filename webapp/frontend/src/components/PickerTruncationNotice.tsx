import { MAX_PAGE_LIMIT } from "@/api/client";

// The full-list pickers request a single page (``limit``) and treat it as
// "everything". When the underlying set is larger, this warns that entries
// past the window are not selectable here, instead of silently hiding them.
export function PickerTruncationNotice({
  total,
  limit = MAX_PAGE_LIMIT,
}: {
  total: number;
  limit?: number;
}) {
  if (total <= limit) return null;
  return (
    <p className="text-xs text-amber-600 dark:text-amber-400">
      Showing the first {limit} of {total}. Entries beyond that are not selectable here — open the
      full list to reach them.
    </p>
  );
}
