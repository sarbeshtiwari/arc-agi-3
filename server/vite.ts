import type { Express } from "express";
import type { Server } from "http";
import type { ViteDevServer } from "vite";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export async function setupVite(app: Express, server: Server) {
  const { createServer: createViteServer } = await import("vite");

  const viteConfig = (await import("../vite.config.js")).default;
  const resolvedConfig = typeof viteConfig === "function" ? await viteConfig() : viteConfig;

  const vite: ViteDevServer = await createViteServer({
    ...resolvedConfig,
    configFile: false,
    server: {
      middlewareMode: true,
      hmr: { server },
    },
    appType: "custom",
  });

  app.use(vite.middlewares);

  // SPA fallback - skip /api routes
  app.use("*", async (req, res, next) => {
    const url = req.originalUrl;
    if (url.startsWith("/api")) return next();

    try {
      const clientDir = path.resolve(__dirname, "../client");
      let html = fs.readFileSync(path.join(clientDir, "index.html"), "utf-8");

      // Cache-bust main entry
      html = html.replace(
        `src="/src/main.tsx"`,
        `src="/src/main.tsx?t=${Date.now()}"`
      );

      html = await vite.transformIndexHtml(url, html);
      res.status(200).set({ "Content-Type": "text/html" }).end(html);
    } catch (e: any) {
      vite.ssrFixStacktrace(e);
      next(e);
    }
  });
}
