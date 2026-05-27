import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StrategyParamsEditor } from "@/components/forms/StrategyParamsEditor";
import type { StrategySchema } from "@/api/strategies";

const SCHEMA: StrategySchema = {
  name: "Demo",
  qualname: "src.strategies.demo.Demo",
  params: [
    { name: "window", kind: "int", default: 20, required: false, nullable: false, choices: null },
    { name: "k", kind: "float", default: 2, required: false, nullable: false, choices: null },
    { name: "label", kind: "str", default: null, required: true, nullable: false, choices: null },
    {
      name: "verbose",
      kind: "bool",
      default: false,
      required: false,
      nullable: false,
      choices: null,
    },
    {
      name: "interval",
      kind: "enum",
      default: "daily",
      required: false,
      nullable: false,
      choices: ["daily", "hour"],
    },
    {
      name: "device",
      kind: "enum",
      default: null,
      required: false,
      nullable: true,
      choices: ["cpu", "cuda", "mps"],
    },
    {
      name: "mode",
      kind: "enum",
      default: null,
      required: true,
      nullable: false,
      choices: ["fast", "slow"],
    },
    {
      name: "weights",
      kind: "complex",
      default: "[]",
      required: false,
      nullable: false,
      choices: null,
    },
    {
      name: "feature_columns",
      kind: "str_list",
      default: null,
      required: true,
      nullable: false,
      choices: null,
    },
    {
      name: "feature_tickers",
      kind: "str_list",
      default: null,
      required: true,
      nullable: false,
      choices: null,
    },
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
    expect(document.querySelector("textarea")).not.toBeNull();
  });

  it("dispatches onChange with parsed numeric value for int params", () => {
    const onChange = vi.fn();
    render(<StrategyParamsEditor schema={SCHEMA} values={{}} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText(/window/i), { target: { value: "42" } });

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ window: 42 }));
  });

  it("labels the empty option per param: required → 'select', nullable → 'none', else 'use default'", () => {
    render(<StrategyParamsEditor schema={SCHEMA} values={{}} onChange={() => undefined} />);
    const intervalSelect = screen.getByRole("combobox", { name: /interval/i });
    expect(intervalSelect.querySelector('option[value=""]')).toHaveTextContent(/use default/i);
    const deviceSelect = screen.getByRole("combobox", { name: /device/i });
    expect(deviceSelect.querySelector('option[value=""]')).toHaveTextContent(/none/i);
    const modeSelect = screen.getByRole("combobox", { name: /mode/i });
    expect(modeSelect.querySelector('option[value=""]')).toHaveTextContent(/select/i);
    expect(modeSelect).toBeRequired();
  });

  it("str_list renders a comma-separated text input and parses on change", () => {
    const onChange = vi.fn();
    render(<StrategyParamsEditor schema={SCHEMA} values={{}} onChange={onChange} />);
    const input = screen.getByLabelText(/feature_columns/i);
    expect(input).toHaveAttribute("type", "text");
    fireEvent.change(input, { target: { value: "rsi_14, macd_signal ma_ratio" } });
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ feature_columns: ["rsi_14", "macd_signal", "ma_ratio"] }),
    );
  });

  it("str_list placeholders are param-name aware (tickers vs columns)", () => {
    render(<StrategyParamsEditor schema={SCHEMA} values={{}} onChange={() => undefined} />);
    expect(screen.getByLabelText(/feature_columns/i)).toHaveAttribute(
      "placeholder",
      expect.stringMatching(/rsi_14/i),
    );
    expect(screen.getByLabelText(/feature_tickers/i)).toHaveAttribute(
      "placeholder",
      expect.stringMatching(/QQQ/),
    );
  });

  it("JSON cell preserves partial text and only commits valid JSON to parent", () => {
    const onChange = vi.fn();
    render(<StrategyParamsEditor schema={SCHEMA} values={{}} onChange={onChange} />);
    const textarea = document.querySelector("textarea");
    if (!textarea) throw new Error("complex param textarea missing");
    fireEvent.change(textarea, { target: { value: '["foo",' } });
    expect(textarea).toHaveValue('["foo",');
    expect(screen.getByText(/JSON:/)).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.change(textarea, { target: { value: '["foo","bar"]' } });
    expect(textarea).toHaveValue('["foo","bar"]');
    expect(screen.queryByText(/JSON:/)).not.toBeInTheDocument();
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ weights: ["foo", "bar"] }));
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
