import type { ReactElement } from "react";

interface AllUsersToggleProps {
  /** Hide the toggle when ``false`` — non-admins never see this control. */
  isAdmin: boolean;
  checked: boolean;
  onChange: (value: boolean) => void;
  /** What artifact kind the toggle applies to — surfaced in the label. */
  artifactLabel: string;
  testId?: string;
}

/**
 * Admin-only "show all users" toggle for artifact list pages.
 *
 * Mirrors the JobsPage pattern: hidden for non-admins, wired to the
 * `?all=true` query param on the matching backend endpoint when checked.
 * The current admin still sees their own + ownerless artifacts when
 * unchecked; the toggle only adds *other users'* artifacts.
 */
export function AllUsersToggle({
  isAdmin,
  checked,
  onChange,
  artifactLabel,
  testId,
}: AllUsersToggleProps): ReactElement | null {
  if (!isAdmin) return null;
  return (
    <label className="flex items-center gap-2 text-sm" data-testid={testId}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => {
          onChange(e.target.checked);
        }}
      />
      Show {artifactLabel} from all users (admin)
    </label>
  );
}
