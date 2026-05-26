import { Link, useLocation } from "react-router-dom";

interface BackLinkProps {
  to: string;
  children: React.ReactNode;
}

export function BackLink({ to, children }: BackLinkProps) {
  // If we arrived via a list-page link that stashed `from` (including
  // sort/pagination/filter query params), prefer it over the bare ``to``
  // so the user returns to the exact list view they came from.
  const location = useLocation();
  const stateFrom =
    location.state &&
    typeof location.state === "object" &&
    "from" in location.state &&
    typeof (location.state as { from: unknown }).from === "string"
      ? (location.state as { from: string }).from
      : null;
  const target = stateFrom ?? to;
  return (
    <div>
      <Link to={target} className="text-xs text-primary hover:underline">
        ← {children}
      </Link>
    </div>
  );
}
