import path from "path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  base: "/",
  build: {
    outDir: path.resolve(__dirname, "../src/team_memory/web/static/dist"),
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:9111",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:9111",
        changeOrigin: true,
      },
      "/ready": {
        target: "http://127.0.0.1:9111",
        changeOrigin: true,
      },
    },
  },
});
