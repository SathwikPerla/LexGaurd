import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const clientId = request.headers.get("x-client-id") ?? "anonymous";
    const res = await fetch(`${BACKEND_URL}/history/${params.id}`, {
      headers: { "X-Client-ID": clientId },
      signal: AbortSignal.timeout(10_000),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err: unknown) {
    return NextResponse.json({ detail: String(err) }, { status: 502 });
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const clientId = request.headers.get("x-client-id") ?? "anonymous";
    const res = await fetch(`${BACKEND_URL}/history/${params.id}`, {
      method: "DELETE",
      headers: { "X-Client-ID": clientId },
      signal: AbortSignal.timeout(10_000),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err: unknown) {
    return NextResponse.json({ detail: String(err) }, { status: 502 });
  }
}
