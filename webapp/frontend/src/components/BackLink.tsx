import { Link } from "react-router-dom";

interface BackLinkProps {
  to: string;
  children: React.ReactNode;
}

export function BackLink({ to, children }: BackLinkProps) {
  return (
    <div>
      <Link to={to} className="text-xs text-primary hover:underline">
        ← {children}
      </Link>
    </div>
  );
}
