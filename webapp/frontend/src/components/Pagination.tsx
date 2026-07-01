import { Button } from "@/components/ui/button";

export interface PaginationProps {
  total: number;
  limit: number;
  offset: number;
  onOffset: (offset: number) => void;
}

export function Pagination({ total, limit, offset, onOffset }: PaginationProps) {
  // Nothing to page through on an empty first page; an out-of-range offset
  // (offset > 0, e.g. paged past a shrunken total) still renders so Previous
  // can walk back into range.
  if (total === 0 && offset === 0) return null;

  // Rows on the current page are a pure slice of the filtered set, so the count
  // is derivable: a short final page (or an offset past the total) yields fewer
  // than ``limit`` and disables Next.
  const count = Math.max(0, Math.min(limit, total - offset));
  // An offset past the total (e.g. a bookmarked deep page after rows were
  // deleted) yields count 0; show "0-0" rather than a reversed "offset+1-total"
  // range. Previous still walks back into range.
  const start = count === 0 ? 0 : offset + 1;
  const end = count === 0 ? 0 : offset + count;
  const hasPrev = offset > 0;
  const hasNext = offset + count < total;

  return (
    <div className="flex items-center justify-between text-sm text-muted-foreground">
      <span>
        Showing {start}-{end} of {total}
      </span>
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={!hasPrev}
          onClick={() => {
            onOffset(Math.max(0, offset - limit));
          }}
        >
          Previous
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={!hasNext}
          onClick={() => {
            onOffset(offset + limit);
          }}
        >
          Next
        </Button>
      </div>
    </div>
  );
}
