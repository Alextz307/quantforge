import type { ReactNode } from "react";
import { Loader2, Minus, TrendingDown, TrendingUp } from "lucide-react";
import { cn } from "@/lib/cn";
import type { SignalKind } from "@/lib/signalKind";

interface SignalBadgeProps {
  signal: number | null;
  kind?: SignalKind;
  loading?: boolean;
  className?: string | undefined;
}

type SignalState = "long" | "short" | "flat" | "unknown" | "computing";

interface SignalDescriptor {
  state: SignalState;
  label: string;
  icon: ReactNode;
  title?: string;
}

const PILL_BASE =
  "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset";

const MUTED_PILL =
  "bg-slate-100 text-slate-600 ring-slate-200 dark:bg-slate-800/60 dark:text-slate-300 dark:ring-slate-700";

const STATE_STYLE: Record<SignalState, string> = {
  long: "bg-emerald-100 text-emerald-800 ring-emerald-200 dark:bg-emerald-900/40 dark:text-emerald-200 dark:ring-emerald-800/60",
  short:
    "bg-rose-100 text-rose-800 ring-rose-200 dark:bg-rose-900/40 dark:text-rose-200 dark:ring-rose-800/60",
  flat: "bg-slate-100 text-slate-700 ring-slate-200 dark:bg-slate-800/60 dark:text-slate-200 dark:ring-slate-700",
  unknown: MUTED_PILL,
  computing: MUTED_PILL,
};

const ICON_CLASS = "h-3 w-3";

function describe(signal: number | null, loading: boolean, kind: SignalKind): SignalDescriptor {
  if (loading) {
    return {
      state: "computing",
      label: "computing...",
      icon: <Loader2 className={cn(ICON_CLASS, "animate-spin")} />,
    };
  }
  if (signal === null || !Number.isFinite(signal)) {
    return { state: "unknown", label: "-", icon: <Minus className={ICON_CLASS} /> };
  }
  if (signal === 0) {
    return {
      state: "flat",
      label: "FLAT",
      icon: <Minus className={ICON_CLASS} />,
      title: signal.toFixed(4),
    };
  }
  // Directional strategies emit +/-1, so the magnitude carries no information -
  // show the word alone. Leverage strategies emit a position-size multiplier,
  // so surface it explicitly as "1.39x".
  const word = signal > 0 ? "LONG" : "SHORT";
  const label = kind === "leverage" ? `${word} | ${Math.abs(signal).toFixed(2)}x` : word;
  return {
    state: signal > 0 ? "long" : "short",
    label,
    icon:
      signal > 0 ? (
        <TrendingUp className={ICON_CLASS} />
      ) : (
        <TrendingDown className={ICON_CLASS} />
      ),
    title: signal.toFixed(4),
  };
}

export function SignalBadge({
  signal,
  kind = "directional",
  loading = false,
  className,
}: SignalBadgeProps) {
  const { state, label, icon, title } = describe(signal, loading, kind);

  return (
    <span
      data-testid="signal-badge"
      data-signal-state={state}
      title={title}
      className={cn(PILL_BASE, STATE_STYLE[state], className)}
    >
      {icon}
      {label}
    </span>
  );
}
