import { z } from "zod";

// Mirrored from webapp/backend/app/schemas/auth.py — kept in sync via the
// check_webapp_schema_mirror.py drift guard. Forms below import these to
// set HTML maxLength so the input stops accepting keystrokes at the limit.
export const LOGIN_USERNAME_MAX = 64;
export const LOGIN_PASSWORD_MAX = 256;

export const loginSchema = z.object({
  username: z
    .string()
    .min(1, "Username is required")
    .max(LOGIN_USERNAME_MAX, `Username must be at most ${String(LOGIN_USERNAME_MAX)} characters`),
  password: z
    .string()
    .min(1, "Password is required")
    .max(LOGIN_PASSWORD_MAX, `Password must be at most ${String(LOGIN_PASSWORD_MAX)} characters`),
});

export type LoginFormValues = z.infer<typeof loginSchema>;
