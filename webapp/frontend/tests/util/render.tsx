import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderOptions } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { ThemeProvider } from "@/lib/theme";
import { ROUTER_FUTURE_FLAGS } from "./router";

interface ProviderOptions {
  initialEntries?: string[];
}

function makeClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, staleTime: 0 },
      mutations: { retry: false },
    },
  });
}

export function renderWithProviders(
  ui: ReactElement,
  { initialEntries = ["/"], ...options }: ProviderOptions & RenderOptions = {},
) {
  const client = makeClient();
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <ThemeProvider>
        <QueryClientProvider client={client}>
          <MemoryRouter initialEntries={initialEntries} future={ROUTER_FUTURE_FLAGS}>
            {children}
          </MemoryRouter>
        </QueryClientProvider>
      </ThemeProvider>
    );
  }
  return { ...render(ui, { wrapper: Wrapper, ...options }), queryClient: client };
}
