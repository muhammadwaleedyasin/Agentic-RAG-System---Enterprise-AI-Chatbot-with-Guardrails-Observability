/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  experimental: {
    instrumentationHook: true,
  },
  async headers() {
    return [
      {
        source: "/api/:path*",
        headers: [
          { key: "Access-Control-Allow-Credentials", value: "true" },
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Access-Control-Allow-Methods", value: "GET,OPTIONS,PATCH,DELETE,POST,PUT" },
          { key: "Access-Control-Allow-Headers", value: "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization" },
        ]
      }
    ]
  },
  async rewrites() {
    return [
      {
        source: '/api/public/:path*',
        destination: '/api/public/:path*',
      },
    ]
  },
  env: {
    DATABASE_URL: process.env.DATABASE_URL,
    NEXTAUTH_SECRET: process.env.NEXTAUTH_SECRET,
    SALT: process.env.SALT,
    ENCRYPTION_KEY: process.env.ENCRYPTION_KEY,
    NEXTAUTH_URL: process.env.NEXTAUTH_URL,
    TELEMETRY_ENABLED: process.env.TELEMETRY_ENABLED,
    LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: process.env.LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES,
  },
}

module.exports = nextConfig