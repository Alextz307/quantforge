import { z } from "zod";
import { ROLES } from "@/api/users";

// Mirrored from webapp/backend/app/schemas/users.py — kept in sync via the
// check_webapp_schema_mirror.py drift guard. Forms below import these to
// set HTML maxLength so the input stops accepting keystrokes at the limit.
export const USER_USERNAME_MAX = 64;
export const USER_PASSWORD_MIN = 8;
export const USER_PASSWORD_MAX = 256;

export const userCreateSchema = z.object({
  username: z
    .string()
    .min(1, "Username is required")
    .max(USER_USERNAME_MAX, `Username must be at most ${String(USER_USERNAME_MAX)} characters`),
  password: z
    .string()
    .min(USER_PASSWORD_MIN, `Password must be at least ${String(USER_PASSWORD_MIN)} characters`)
    .max(USER_PASSWORD_MAX, `Password must be at most ${String(USER_PASSWORD_MAX)} characters`),
  role: z.enum(ROLES).default("user"),
});

export type UserCreateFormValues = z.infer<typeof userCreateSchema>;
