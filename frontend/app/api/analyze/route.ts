import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  try {
    // Forward the raw request body directly to the backend without parsing.
    // This avoids Next.js body-size limits and multipart parsing failures.
    const contentType = request.headers.get("content-type") ?? "";

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const response = await fetch(`${BACKEND_URL}/analyze`, {
      method: "POST",
      headers: {
        "content-type": contentType,
        "x-client-id": request.headers.get("x-client-id") ?? "anonymous",
      },
      body: request.body,
      signal: AbortSignal.timeout(300_000),
      ...(({ duplex: "half" }) as Record<string, unknown>),
    } as RequestInit);

    const data = await response.json().catch(() => ({
      detail: `Backend returned status ${response.status} with non-JSON body`,
    }));

    return NextResponse.json(data, { status: response.status });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[/api/analyze]", msg);
    return NextResponse.json(
      { detail: `Proxy error: ${msg}` },
      { status: 502 }
    );
  }
}
