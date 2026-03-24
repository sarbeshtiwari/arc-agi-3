import type { Request, Response, NextFunction } from "express";

// Wraps async route handlers to catch errors and pass to next()
export function asyncHandler(fn: (req: Request, res: Response, next: NextFunction) => Promise<any>) {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}
