import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { YamlEditor } from "@/components/YamlEditor";
import { renderWithProviders } from "../util/render";

describe("YamlEditor", () => {
  it("renders the value in the editor textarea", () => {
    renderWithProviders(<YamlEditor value="name: hello" onChange={() => undefined} />);
    const editor = screen.getByTestId<HTMLTextAreaElement>("monaco-editor");
    expect(editor.value).toBe("name: hello");
  });

  it("forwards edits to onChange", () => {
    const handle = vi.fn();
    renderWithProviders(<YamlEditor value="" onChange={handle} />);
    const editor = screen.getByTestId("monaco-editor");
    fireEvent.change(editor, { target: { value: "name: new" } });
    expect(handle).toHaveBeenCalledWith("name: new");
  });

  it("respects readOnly", () => {
    renderWithProviders(<YamlEditor value="x" onChange={() => undefined} readOnly />);
    const editor = screen.getByTestId<HTMLTextAreaElement>("monaco-editor");
    expect(editor.readOnly).toBe(true);
  });
});
