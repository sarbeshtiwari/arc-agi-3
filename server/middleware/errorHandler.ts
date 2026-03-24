import type { Request, Response, NextFunction } from "express";

export class AppError extends Error {
  statusCode: number;
  constructor(message: string, statusCode: number = 500) {
    super(message);
    this.statusCode = statusCode;
    this.name = "AppError";
  }
}

export function errorHandler(err: any, req: Request, res: Response, next: NextFunction) {
  console.error(`[ERROR] ${req.method} ${req.path}:`, err.message);

  if (err instanceof AppError) {
    return res.status(err.statusCode).json({ detail: err.message });
  }

  const status = err.statusCode || err.status || 500;
  const message = err.message || "Internal server error";
  res.status(status).json({ detail: message });
}
