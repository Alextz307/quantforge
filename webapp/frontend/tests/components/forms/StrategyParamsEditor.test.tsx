import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StrategyParamsEditor } from "@/components/forms/StrategyParamsEditor";
import type { StrategySchema } from "@/api/strategies";

const SCHEMA: StrategySchema = {
  name: "Demo",
  qualname: "src.strategies.demo.Demo",
  params: [
    { name: "window", kind: "int", default: 20, required: false, choices: null },
    { name: "k", kind: "float", default: 2, required: false, choices: null },
    { name: "label", kind: "str", default: null, required: true, choices: null },
    { name: "verbose", kind: "bool", default: false, required: false, choices: null },
    {
      name: "interval",
      kind: "enum",
      default: "daily",
      required: false,
      choices: ["daily", "hour"],
    },
    { name: "weights", kind: "complex", default: "[]", required: false, choices: null },
  ],
};

const EMPTY_SCHEMA: StrategySchema = { name: "Empty", qualname: "x", params: [] };

describe("StrategyParamsEditor", () => {
  it("renders the empty-state message when the strategy takes no params", () => {
    render(<StrategyParamsEditor schema={EMPTY_SCHEMA} values={{}} onChange={() => undefined} />);
    expect(screen.getByText(/no constructor parameters/i)).toBeInTheDocument();
  });

  it("renders an input for each ParamKind including the JSON editor for complex", () => {
    render(<StrategyParamsEditor schema={SCHEMA} values={{}} onChange={() => undefined} />);
    expect(screen.getByLabelText(/window/i)).toHaveAttribute("type", "number");
    expect(screen.getByLabelText(/label/i)).toHaveAttribute("type", "text");
    expect(screen.getByLabelText(/verbose/i)).toHaveAttribute("type", "checkbox");
    expect(screen.getByRole("combobox", { name: /interval/i })).toBeInTheDocument();
    // Complex params render as <textarea>; matched by name via the for/id pairing.
    expect(document.querySelector("textarea")).not.toBeNull();
  });

  it("dispatches onChange with parsed numeric value for int params", () => {
    const onChange = vi.fn();
    render(<StrategyParamsEditor schema={SCHEMA} values={{}} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText(/window/i), { target: { value: "42" } });

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ window: 42 }));
  });

  it("surfaces server-side errors via errorsByLoc", () => {
    const errors = new Map([["strategy.params.window", "must be greater than 0"]]);
    render(
      <StrategyParamsEditor
        schema={SCHEMA}
        values={{}}
        onChange={() => undefined}
        errorsByLoc={errors}
      />,
    );
    expect(screen.getByText(/must be greater than 0/i)).toBeInTheDocument();
  });
});
