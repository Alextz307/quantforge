import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/lib/theme";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  const next = isDark ? "light" : "dark";
  const label = `Switch to ${next} theme`;
  const Icon = isDark ? Moon : Sun;
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      data-testid="theme-toggle"
      data-theme={resolvedTheme}
      aria-label={label}
      title={label}
      onClick={() => {
        setTheme(next);
      }}
    >
      <Icon className="h-4 w-4" />
    </Button>
  );
}
