export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:8000";

export class APIError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "APIError";
  }
}

export async function fetchAPI<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        ...(options.body && !(options.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new APIError(response.status, errorText || `HTTP Error ${response.status}`);
    }

    // Some endpoints might return empty responses, handle safely
    const text = await response.text();
    if (!text) return {} as T;
    
    return JSON.parse(text) as T;
  } catch (error) {
    if (error instanceof APIError) throw error;
    throw new Error(error instanceof Error ? error.message : "Network error");
  }
}
