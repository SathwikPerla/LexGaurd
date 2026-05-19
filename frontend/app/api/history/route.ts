import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function GET(request: NextRequest) {
  try {
    const clientId = request.headers.get("x-client-id") ?? "anonymous";
    const res = await fetch(`${BACKEND_URL}/history`, {
      headers: { "X-Client-ID": clientId },
      signal: AbortSignal.timeout(10_000),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err: unknown) {
    return NextResponse.json({ detail: String(err) }, { status: 502 });
  }
}
