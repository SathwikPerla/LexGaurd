/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    // BACKEND_URL is a server-side env var — read at server startup, not baked
    // into the client bundle. Works correctly in both dev and production.
    // Local:      set in frontend/.env.local
    // Production: set in Vercel dashboard as BACKEND_URL
    const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
