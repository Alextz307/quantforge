import { zodResolver } from "@hookform/resolvers/zod";
import { Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { useForm } from "react-hook-form";
import {
  ROLE_USER,
  ROLES,
  useCreateUser,
  useDeleteUser,
  useUsers,
  type UserCreate,
  type UserPublic,
} from "@/api/users";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  USER_PASSWORD_MAX,
  USER_USERNAME_MAX,
  userCreateSchema,
  type UserCreateFormValues,
} from "@/lib/schemas/userCreate";

const EMPTY_FORM: UserCreateFormValues = { username: "", password: "", role: ROLE_USER };

export function UserList() {
  const usersQuery = useUsers();
  const createUser = useCreateUser();
  const deleteUser = useDeleteUser();
  const [showAutoCreatedOnly, setShowAutoCreatedOnly] = useState(false);
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<UserCreateFormValues>({
    resolver: zodResolver(userCreateSchema),
    defaultValues: EMPTY_FORM,
  });

  const onSubmit = handleSubmit((values) => {
    createUser.mutate(values satisfies UserCreate, {
      onSuccess: () => {
        reset(EMPTY_FORM);
      },
    });
  });

  const submitDisabled = isSubmitting || createUser.isPending;

  const filteredUsers = useMemo(() => {
    if (!usersQuery.data) return undefined;
    if (!showAutoCreatedOnly) return usersQuery.data;
    return usersQuery.data.filter((u) => u.auto_created_at !== null);
  }, [usersQuery.data, showAutoCreatedOnly]);

  const onDeleteClick = (user: UserPublic) => {
    if (!window.confirm(`Soft-delete user "${user.username}" (id=${String(user.id)})?`)) return;
    deleteUser.mutate(user.id);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Add user</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} noValidate className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="new-username">Username</Label>
                <Input id="new-username" maxLength={USER_USERNAME_MAX} {...register("username")} />
                {errors.username && (
                  <p className="text-sm text-destructive">{errors.username.message}</p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="new-password">Password</Label>
                <Input
                  id="new-password"
                  type="password"
                  maxLength={USER_PASSWORD_MAX}
                  {...register("password")}
                />
                {errors.password ? (
                  <p className="text-sm text-destructive">{errors.password.message}</p>
                ) : (
                  <p className="text-xs text-muted-foreground">At least 8 characters.</p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="new-role">Role</Label>
                <select
                  id="new-role"
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  {...register("role")}
                >
                  {ROLES.map((role) => (
                    <option key={role} value={role}>
                      {role}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="flex justify-end">
              <Button type="submit" disabled={submitDisabled}>
                Create
              </Button>
            </div>
          </form>
          {createUser.isError && (
            <Alert variant="destructive" className="mt-4">
              <AlertDescription>{createUser.error.message}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-4">
          <CardTitle>Users</CardTitle>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={showAutoCreatedOnly}
              onChange={(e) => {
                setShowAutoCreatedOnly(e.target.checked);
              }}
              data-testid="users-auto-created-toggle"
            />
            Show only auto-created (CLI typos)
          </label>
        </CardHeader>
        <CardContent className="space-y-4">
          {usersQuery.isLoading && <p className="text-sm text-muted-foreground">Loading...</p>}
          {usersQuery.isError && (
            <Alert variant="destructive">
              <AlertDescription>{usersQuery.error.message}</AlertDescription>
            </Alert>
          )}
          {deleteUser.isError && (
            <Alert variant="destructive">
              <AlertDescription>{deleteUser.error.message}</AlertDescription>
            </Alert>
          )}
          {filteredUsers && (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-muted-foreground">
                  <th className="py-2">Username</th>
                  <th className="py-2">Role</th>
                  <th className="py-2">Auto-created</th>
                  <th className="py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map((user) => (
                  <tr key={user.id} className="border-b last:border-0">
                    <td className="py-2">{user.username}</td>
                    <td className="py-2">{user.role}</td>
                    <td className="py-2 font-mono text-xs">
                      {user.auto_created_at ?? (
                        <span className="text-muted-foreground italic">-</span>
                      )}
                    </td>
                    <td className="py-2 text-right">
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={`Delete ${user.username}`}
                        disabled={deleteUser.isPending}
                        onClick={() => {
                          onDeleteClick(user);
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </td>
                  </tr>
                ))}
                {filteredUsers.length === 0 && (
                  <tr>
                    <td colSpan={4} className="py-3 text-sm text-muted-foreground">
                      {showAutoCreatedOnly
                        ? "No auto-created accounts - nothing to clean up."
                        : "No users."}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
