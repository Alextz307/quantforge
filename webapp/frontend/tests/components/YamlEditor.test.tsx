import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { YamlEditor } from "@/components/YamlEditor";

describe("YamlEditor", () => {
  it("renders the value in the editor textarea", () => {
    render(<YamlEditor value="name: hello" onChange={() => undefined} />);
    const editor = screen.getByTestId<HTMLTextAreaElement>("monaco-editor");
    expect(editor.value).toBe("name: hello");
  });

  it("forwards edits to onChange", () => {
    const handle = vi.fn();
    render(<YamlEditor value="" onChange={handle} />);
    const editor = screen.getByTestId("monaco-editor");
    fireEvent.change(editor, { target: { value: "name: new" } });
    expect(handle).toHaveBeenCalledWith("name: new");
  });

  it("respects readOnly", () => {
    render(<YamlEditor value="x" onChange={() => undefined} readOnly />);
    const editor = screen.getByTestId<HTMLTextAreaElement>("monaco-editor");
    expect(editor.readOnly).toBe(true);
  });
});
