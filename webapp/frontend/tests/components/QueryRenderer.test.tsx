import { render, screen } from "@testing-library/react";
import type { UseQueryResult } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";
import { QueryRenderer } from "@/components/QueryRenderer";

function pending(): UseQueryResult<string> {
  return { isPending: true, isError: false, data: undefined } as unknown as UseQueryResult<string>;
}
function failure(message: string): UseQueryResult<string> {
  return {
    isPending: false,
    isError: true,
    error: new Error(message),
    data: undefined,
  } as unknown as UseQueryResult<string>;
}
function success(value: string): UseQueryResult<string> {
  return { isPending: false, isError: false, data: value } as unknown as UseQueryResult<string>;
}

describe("QueryRenderer", () => {
  it("renders the loading message while the query is pending", () => {
    render(
      <QueryRenderer query={pending()} errorTitle="X" loadingMessage="Fetching things...">
        {() => <span>data</span>}
      </QueryRenderer>,
    );
    expect(screen.getByText("Fetching things...")).toBeInTheDocument();
    expect(screen.queryByText("data")).not.toBeInTheDocument();
  });

  it("surfaces the error title and message when the query fails", () => {
    render(
      <QueryRenderer query={failure("kaboom")} errorTitle="Failed to load X">
        {() => <span>data</span>}
      </QueryRenderer>,
    );
    expect(screen.getByText("Failed to load X")).toBeInTheDocument();
    expect(screen.getByText("kaboom")).toBeInTheDocument();
  });

  it("renders the children with data on success", () => {
    render(
      <QueryRenderer query={success("hello")} errorTitle="X">
        {(data) => <span>got: {data}</span>}
      </QueryRenderer>,
    );
    expect(screen.getByText("got: hello")).toBeInTheDocument();
  });
});
