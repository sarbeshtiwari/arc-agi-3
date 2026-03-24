import "dotenv/config";
import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import path from "path";
import { fileURLToPath } from "url";
import { registerRoutes } from "./routes.js";
import { setupVite } from "./vite.js";

const app = express();
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Middleware
app.use(cors({ origin: true, credentials: true }));
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));
app.use(cookieParser());
app.set("trust proxy", 1);

// Request logger (simple)
app.use((req, res, next) => {
  const start = Date.now();
  res.on("finish", () => {
    const ms = Date.now() - start;
    if (req.path.startsWith("/api")) {
      console.log(`${req.method} ${req.path} ${res.statusCode} ${ms}ms`);
    }
  });
  next();
});

// Register routes and get HTTP server
const server = await registerRoutes(app);

// Setup Vite (dev) or static serving (prod)
const isDev = process.env.NODE_ENV !== "production";
if (isDev) {
  await setupVite(app, server);
} else {
  const distPath = path.resolve(__dirname, "../dist/public");
  app.use(express.static(distPath));
  app.get("*", (req, res) => {
    if (!req.path.startsWith("/api")) {
      res.sendFile(path.join(distPath, "index.html"));
    }
  });
}

// Start
const port = parseInt(process.env.PORT || "5000");
server.listen(port, "0.0.0.0", () => {
  console.log(`[SERVER] Running on http://localhost:${port} (${isDev ? "development" : "production"})`);
});
