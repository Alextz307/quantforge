import { z } from "zod";

export const loginSchema = z.object({
  username: z
    .string()
    .min(1, "Username is required")
    .max(64, "Username must be at most 64 characters"),
  password: z
    .string()
    .min(1, "Password is required")
    .max(256, "Password must be at most 256 characters"),
});

export type LoginFormValues = z.infer<typeof loginSchema>;
