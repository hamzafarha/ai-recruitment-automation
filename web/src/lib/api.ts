// The FastAPI backend (same origin: served by it in production, proxied by Vite in development).
export const API_PREFIX = "/api/v1";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function detailOf(body: any, fallback: string): string {
  const detail = body?.detail ?? body;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail))
    return detail.map((d: any) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
  return detail ? JSON.stringify(detail) : fallback;
}

async function request<T = any>(method: string, path: string, body?: unknown, prefix = API_PREFIX): Promise<T> {
  let response: Response;
  const isForm = body instanceof FormData;
  try {
    response = await fetch(prefix + path, {
      method,
      headers: body !== undefined && !isForm ? { "Content-Type": "application/json" } : undefined,
      body: body === undefined ? undefined : isForm ? (body as FormData) : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "The API is not reachable. Start it: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000");
  }
  if (!response.ok) {
    let parsed: any = null;
    try {
      parsed = await response.json();
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, detailOf(parsed, `${response.status} ${response.statusText}`));
  }
  const type = response.headers.get("content-type") || "";
  return (type.includes("json") ? response.json() : response.text()) as Promise<T>;
}

export const api = {
  get: <T = any>(path: string) => request<T>("GET", path),
  post: <T = any>(path: string, body: unknown = {}) => request<T>("POST", path, body),
  patch: <T = any>(path: string, body: unknown) => request<T>("PATCH", path, body),
  del: <T = any>(path: string) => request<T>("DELETE", path),
  upload: <T = any>(path: string, form: FormData) => request<T>("POST", path, form),
};

export const getHealth = () => request("GET", "/health", undefined, API_PREFIX);

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] ?? "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

export const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));
