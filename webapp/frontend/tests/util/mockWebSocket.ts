import { afterAll, afterEach, beforeAll } from "vitest";

export class MockWebSocket {
  static instances: MockWebSocket[] = [];
  url: string;
  readyState = 0;
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent<string>) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }
  triggerOpen(): void {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }
  triggerMessage(payload: object): void {
    this.onmessage?.(new MessageEvent("message", { data: JSON.stringify(payload) }));
  }
  triggerClose(): void {
    this.readyState = 3;
    this.onclose?.(new CloseEvent("close"));
  }
  close(): void {
    this.triggerClose();
  }
}

export function installMockWebSocket(): void {
  const real = globalThis.WebSocket;
  beforeAll(() => {
    Object.defineProperty(globalThis, "WebSocket", {
      value: MockWebSocket,
      writable: true,
      configurable: true,
    });
  });
  afterAll(() => {
    Object.defineProperty(globalThis, "WebSocket", {
      value: real,
      writable: true,
      configurable: true,
    });
  });
  afterEach(() => {
    MockWebSocket.instances = [];
  });
}
