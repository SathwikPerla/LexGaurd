import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.text();

    const response = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-client-id": request.headers.get("x-client-id") ?? "anonymous",
      },
      body,
      signal: AbortSignal.timeout(200_000),
    });

    const data = await response.json().catch(() => ({
      detail: `Backend returned status ${response.status} with non-JSON body`,
    }));

    return NextResponse.json(data, { status: response.status });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[/api/chat]", msg);
    return NextResponse.json({ detail: `Proxy error: ${msg}` }, { status: 502 });
  }
}
