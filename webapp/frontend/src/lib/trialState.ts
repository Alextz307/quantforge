export type TrialState = "COMPLETE" | "FAIL" | "PRUNED" | "RUNNING";

export const TRIAL_STATE_COMPLETE: TrialState = "COMPLETE";

export const TRIAL_STATE_STYLES: Record<TrialState, string> = {
  COMPLETE: "bg-green-100 text-green-900 dark:bg-green-900/40 dark:text-green-200",
  FAIL: "bg-red-100 text-red-900 dark:bg-red-900/40 dark:text-red-200",
  PRUNED: "bg-yellow-100 text-yellow-900 dark:bg-yellow-900/40 dark:text-yellow-200",
  RUNNING: "bg-blue-100 text-blue-900 dark:bg-blue-900/40 dark:text-blue-200",
};

export const TRIAL_STATE_FALLBACK_STYLE = "bg-muted text-muted-foreground";

export function trialStateStyle(state: string): string {
  return (TRIAL_STATE_STYLES as Record<string, string>)[state] ?? TRIAL_STATE_FALLBACK_STYLE;
}
