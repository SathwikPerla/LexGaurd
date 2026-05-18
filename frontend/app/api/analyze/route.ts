import { NextRequest, NextResponse } from "next/server";

// This Route Handler runs on the Next.js server — no CORS, no IPv6 issues.
// It receives the file from the browser, forwards it to the Python backend,
// and returns the response. Timeout: 5 minutes (pipeline takes 60–120s).
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file");

    if (!file) {
      return NextResponse.json({ detail: "No file attached." }, { status: 422 });
    }

    // Forward the file to the Python backend
    const backendForm = new FormData();
    backendForm.append("file", file as Blob);

    const response = await fetch(`${BACKEND_URL}/analyze`, {
      method: "POST",
      body: backendForm,
      // AbortSignal.timeout available in Node 17.3+ — we're on Node 24
      signal: AbortSignal.timeout(300_000), // 5 minutes
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[/api/analyze] proxy error:", msg);
    return NextResponse.json(
      { detail: `Analysis failed: ${msg}. Is the backend running on port 8000?` },
      { status: 502 }
    );
  }
}
