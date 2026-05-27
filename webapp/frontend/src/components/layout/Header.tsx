import { LogOut } from "lucide-react";
import { useLogout } from "@/api/auth";
import type { UserPublic } from "@/api/users";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";

interface HeaderProps {
  user: UserPublic;
}

export function Header({ user }: HeaderProps) {
  const logout = useLogout();
  return (
    <header className="flex h-16 items-center justify-end border-b bg-card px-6">
      <div className="flex items-center gap-4">
        <ThemeToggle />
        <span className="text-sm text-muted-foreground">
          {user.username}
          <span className="ml-2 rounded-full bg-secondary px-2 py-0.5 text-xs">{user.role}</span>
        </span>
        <Button
          variant="ghost"
          size="sm"
          disabled={logout.isPending}
          onClick={() => {
            logout.mutate();
          }}
        >
          <LogOut className="mr-2 h-4 w-4" />
          Sign out
        </Button>
      </div>
    </header>
  );
}
