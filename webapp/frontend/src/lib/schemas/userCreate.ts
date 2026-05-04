import { z } from "zod";
import { ROLES } from "@/api/users";

export const userCreateSchema = z.object({
  username: z
    .string()
    .min(1, "Username is required")
    .max(64, "Username must be at most 64 characters"),
  password: z
    .string()
    .min(8, "Password must be at least 8 characters")
    .max(256, "Password must be at most 256 characters"),
  role: z.enum(ROLES),
});

export type UserCreateFormValues = z.infer<typeof userCreateSchema>;
