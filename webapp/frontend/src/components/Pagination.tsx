import { Button } from "@/components/ui/button";

export interface PaginationProps {
  total: number;
  limit: number;
  offset: number;
  // Rows actually on the current page; drives the "Showing x-y" range and lets
  // a short final page (count < limit) disable Next once offset + count == total.
  count: number;
  onOffset: (offset: number) => void;
}

export function Pagination({ total, limit, offset, count, onOffset }: PaginationProps) {
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(total, offset + count);
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
