import {
  GenesysClient,
  type BaseMessage,
  type SessionResponse,
  type StructuredMessage,
  type ConnectionClosedEvent,
  type SessionExpiredEvent,
} from "./genesysClient";
import { jest, expect } from "@jest/globals";

class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;

  constructor(public url: string) {}

  send = jest.fn();
  close = jest.fn();

  simulateOpen() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  simulateMessage(data: string) {
    this.onmessage?.({ data } as MessageEvent);
  }

  simulateClose(event?: Partial<CloseEvent>) {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.(event as CloseEvent);
  }

  simulateError() {
    this.onerror?.();
  }
}

let mockWebSocketInstance: MockWebSocket | null = null;

function setMockWebSocketInstance(instance: MockWebSocket) {
  mockWebSocketInstance = instance;
}

// Replace global WebSocket with mock
(global as unknown as { WebSocket: typeof MockWebSocket }).WebSocket =
  class extends MockWebSocket {
    constructor(url: string) {
      super(url);
      setMockWebSocketInstance(this);
    }
  };

describe("GenesysClient", () => {
  let client: GenesysClient;

  beforeEach(() => {
    mockWebSocketInstance = null;
    client = new GenesysClient({
      websocketUrl: "wss://test.example.com/v2/messaging",
      deploymentKey: "test-deployment-id",
    });
  });

  afterEach(() => {
    client.disconnect();
  });

  describe("WebSocket message handlers", () => {
    beforeEach(async () => {
      const connectPromise = client.connect();
      mockWebSocketInstance?.simulateOpen();
      await connectPromise;
    });

    it("should call sessionResponse handler when receiving SessionResponse message", () => {
      const sessionResponseHandler = jest.fn();
      client.on("sessionResponse", sessionResponseHandler);

      const sessionResponseBody: SessionResponse = {
        connected: true,
        newSession: true,
        tracingId: "trace-123",
      };

      const message: BaseMessage = {
        type: "response",
        class: "SessionResponse",
        code: 200,
        body: sessionResponseBody,
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(message));

      expect(sessionResponseHandler).toHaveBeenCalledTimes(1);
      expect(sessionResponseHandler).toHaveBeenCalledWith(sessionResponseBody);
    });

    it("should call message handler when receiving StructuredMessage", () => {
      const messageHandler = jest.fn();
      client.on("message", messageHandler);

      const structuredMessageBody: StructuredMessage = {
        type: "Text",
        text: "Hello from agent",
        direction: "Inbound",
        id: "msg-123",
        channel: {
          from: {
            nickname: "Agent",
          },
        },
      };

      const message: BaseMessage = {
        type: "message",
        class: "StructuredMessage",
        code: 200,
        body: structuredMessageBody,
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(message));

      expect(messageHandler).toHaveBeenCalledTimes(1);
      expect(messageHandler).toHaveBeenCalledWith(structuredMessageBody);
    });

    it("should call connectionClosed handler when receiving ConnectionClosedEvent", () => {
      const connectionClosedHandler = jest.fn();
      client.on("connectionClosed", connectionClosedHandler);

      const connectionClosedBody: ConnectionClosedEvent = {
        tracingId: "trace-456",
        reason: "Session ended by agent",
      };

      const message: BaseMessage = {
        type: "message",
        class: "ConnectionClosedEvent",
        code: 200,
        body: connectionClosedBody,
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(message));

      expect(connectionClosedHandler).toHaveBeenCalledTimes(1);
      expect(connectionClosedHandler).toHaveBeenCalledWith(
        connectionClosedBody,
      );
    });

    it("should call sessionExpired handler when receiving SessionExpiredEvent", () => {
      const sessionExpiredHandler = jest.fn();
      client.on("sessionExpired", sessionExpiredHandler);

      const sessionExpiredBody: SessionExpiredEvent = {
        tracingId: "trace-789",
      };

      const message: BaseMessage = {
        type: "message",
        class: "SessionExpiredEvent",
        code: 200,
        body: sessionExpiredBody,
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(message));

      expect(sessionExpiredHandler).toHaveBeenCalledTimes(1);
      expect(sessionExpiredHandler).toHaveBeenCalledWith(sessionExpiredBody);
    });

    it("should call error handler when receiving Error message", () => {
      const errorHandler = jest.fn();
      client.on("error", errorHandler);

      const errorBody = {
        message: "Something went wrong",
        code: "INVALID_REQUEST",
      };

      const message: BaseMessage = {
        type: "response",
        class: "Error",
        code: 400,
        body: errorBody,
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(message));

      expect(errorHandler).toHaveBeenCalledTimes(1);
      expect(errorHandler).toHaveBeenCalledWith(expect.any(Error));
      expect((errorHandler.mock.calls[0][0] as Error).message).toContain(
        "Something went wrong",
      );
    });

    it("should call rawMessage handler for all incoming messages", () => {
      const rawMessageHandler = jest.fn();
      client.on("rawMessage", rawMessageHandler);

      const message: BaseMessage = {
        type: "response",
        class: "SessionResponse",
        code: 200,
        body: { connected: true },
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(message));

      expect(rawMessageHandler).toHaveBeenCalledTimes(1);
      expect(rawMessageHandler).toHaveBeenCalledWith(message);
    });

    it("should call error handler when receiving invalid JSON", () => {
      const errorHandler = jest.fn();
      client.on("error", errorHandler);

      mockWebSocketInstance?.simulateMessage("not valid json");

      expect(errorHandler).toHaveBeenCalledTimes(1);
      expect(errorHandler).toHaveBeenCalledWith(expect.any(Error));
    });

    it("should call multiple handlers for the same event", () => {
      const handler1 = jest.fn();
      const handler2 = jest.fn();
      client.on("message", handler1);
      client.on("message", handler2);

      const structuredMessageBody: StructuredMessage = {
        type: "Text",
        text: "Test message",
      };

      const message: BaseMessage = {
        type: "message",
        class: "StructuredMessage",
        code: 200,
        body: structuredMessageBody,
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(message));

      expect(handler1).toHaveBeenCalledTimes(1);
      expect(handler2).toHaveBeenCalledTimes(1);
    });

    it("should not call handler after it is removed with off()", () => {
      const messageHandler = jest.fn();
      client.on("message", messageHandler);
      client.off("message", messageHandler);

      const message: BaseMessage = {
        type: "message",
        class: "StructuredMessage",
        code: 200,
        body: { type: "Text", text: "Test" },
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(message));

      expect(messageHandler).not.toHaveBeenCalled();
    });

    it("should call both rawMessage and specific handler for a message", () => {
      const rawMessageHandler = jest.fn();
      const sessionResponseHandler = jest.fn();
      client.on("rawMessage", rawMessageHandler);
      client.on("sessionResponse", sessionResponseHandler);

      const message: BaseMessage = {
        type: "response",
        class: "SessionResponse",
        code: 200,
        body: { connected: true },
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(message));

      expect(rawMessageHandler).toHaveBeenCalledTimes(1);
      expect(sessionResponseHandler).toHaveBeenCalledTimes(1);
    });
  });

  describe("WebSocket connection events", () => {
    it("should call connected handler when WebSocket opens", async () => {
      const connectedHandler = jest.fn();
      client.on("connected", connectedHandler);

      const connectPromise = client.connect();
      mockWebSocketInstance?.simulateOpen();
      await connectPromise;

      expect(connectedHandler).toHaveBeenCalledTimes(1);
    });

    it("should call disconnected handler when WebSocket closes", async () => {
      const disconnectedHandler = jest.fn();
      client.on("disconnected", disconnectedHandler);

      const connectPromise = client.connect();
      mockWebSocketInstance?.simulateOpen();
      await connectPromise;

      const closeEvent = { code: 1000, reason: "Normal closure" };
      mockWebSocketInstance?.simulateClose(closeEvent);

      expect(disconnectedHandler).toHaveBeenCalledTimes(1);
      expect(disconnectedHandler).toHaveBeenCalledWith(closeEvent);
    });

    it("should call error handler and reject promise when WebSocket errors during connect", async () => {
      const errorHandler = jest.fn();
      client.on("error", errorHandler);

      const connectPromise = client.connect();
      mockWebSocketInstance?.simulateError();

      await expect(connectPromise).rejects.toThrow(
        "WebSocket connection error",
      );
      expect(errorHandler).toHaveBeenCalledTimes(1);
    });
  });
});
