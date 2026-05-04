import { UserList } from "@/features/admin/UserList";

export function AdminPage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Admin</h1>
        <p className="text-sm text-muted-foreground">Manage user accounts and roles.</p>
      </div>
      <UserList />
    </div>
  );
}
