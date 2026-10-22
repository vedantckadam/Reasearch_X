let messageCounter = 0;

export function generateMessageId(prefix: "user" | "asst"): string {
  messageCounter += 1;
  return `${prefix}-${messageCounter}`;
}

export function getCurrentTimestamp(): string {
  const now = new Date();
  const hours = now.getHours().toString().padStart(2, "0");
  const minutes = now.getMinutes().toString().padStart(2, "0");
  return `${hours}:${minutes}`;
}
