import pg from "pg";

const { Pool } = pg;

const isSSL = process.env.DATABASE_URL?.includes("neon.tech") ||
  process.env.DATABASE_URL?.includes("sslmode=require");

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 50,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000,
  ssl: isSSL ? { rejectUnauthorized: false } : undefined,
});

pool.on("error", (err) => {
  console.error("[DB] Unexpected pool error:", err.message);
});

// ──── Initialize tables ────
export async function initDB() {
  const client = await pool.connect();
  try {
    await client.query(`
      CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        hashed_password VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE,
        is_admin BOOLEAN DEFAULT FALSE,
        is_active BOOLEAN DEFAULT TRUE,
        allowed_pages JSONB DEFAULT '[]'::jsonb,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
      );

      CREATE TABLE IF NOT EXISTS games (
        id TEXT PRIMARY KEY,
        game_id VARCHAR(50) UNIQUE NOT NULL,
        name VARCHAR(100) NOT NULL,
        description TEXT,
        game_rules TEXT,
        game_owner_name VARCHAR(100),
        game_drive_link VARCHAR(500),
        game_video_link VARCHAR(500),
        version VARCHAR(20) NOT NULL DEFAULT 'v1',
        game_code VARCHAR(10) NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        default_fps INTEGER DEFAULT 5,
        baseline_actions JSONB,
        tags JSONB,
        grid_max_size INTEGER DEFAULT 64,
        game_file_path VARCHAR(500) NOT NULL,
        metadata_file_path VARCHAR(500) NOT NULL,
        local_dir VARCHAR(500) NOT NULL,
        total_plays INTEGER DEFAULT 0,
        total_wins INTEGER DEFAULT 0,
        avg_score REAL DEFAULT 0.0,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        uploaded_by TEXT REFERENCES users(id)
      );

      CREATE TABLE IF NOT EXISTS play_sessions (
        id TEXT PRIMARY KEY,
        game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
        user_id TEXT REFERENCES users(id),
        session_guid VARCHAR(100) UNIQUE NOT NULL,
        seed INTEGER DEFAULT 0,
        current_level INTEGER DEFAULT 0,
        state VARCHAR(20) DEFAULT 'NOT_FINISHED',
        score REAL DEFAULT 0.0,
        total_actions INTEGER DEFAULT 0,
        game_overs INTEGER DEFAULT 0,
        player_name VARCHAR(100),
        total_time REAL DEFAULT 0.0,
        level_stats JSONB DEFAULT '[]'::jsonb,
        action_log JSONB DEFAULT '[]'::jsonb,
        started_at TIMESTAMPTZ DEFAULT NOW(),
        ended_at TIMESTAMPTZ
      );

      CREATE TABLE IF NOT EXISTS game_analytics (
        id TEXT PRIMARY KEY,
        game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
        date TIMESTAMPTZ DEFAULT NOW(),
        plays_count INTEGER DEFAULT 0,
        wins_count INTEGER DEFAULT 0,
        avg_actions_to_win REAL DEFAULT 0.0,
        avg_score REAL DEFAULT 0.0,
        unique_players INTEGER DEFAULT 0
      );

      CREATE TABLE IF NOT EXISTS game_requests (
        id TEXT PRIMARY KEY,
        game_id VARCHAR(50) UNIQUE NOT NULL,
        requester_name VARCHAR(100) NOT NULL,
        requester_email VARCHAR(255),
        message TEXT,
        description TEXT,
        game_rules TEXT,
        game_owner_name VARCHAR(100),
        game_drive_link VARCHAR(500),
        game_video_link VARCHAR(500),
        status VARCHAR(20) DEFAULT 'pending',
        admin_note TEXT,
        game_file_path VARCHAR(500),
        metadata_file_path VARCHAR(500),
        local_dir VARCHAR(500),
        game_file_content BYTEA,
        metadata_file_content BYTEA,
        game_code VARCHAR(50),
        version VARCHAR(20) DEFAULT 'v1',
        tags JSONB,
        default_fps INTEGER DEFAULT 5,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        reviewed_at TIMESTAMPTZ,
        reviewed_by TEXT REFERENCES users(id)
      );

      CREATE TABLE IF NOT EXISTS temp_game_sessions (
        id TEXT PRIMARY KEY,
        session_guid VARCHAR(50) UNIQUE NOT NULL,
        game_id VARCHAR(50),
        player_name VARCHAR(100),
        state VARCHAR(20) DEFAULT 'NOT_FINISHED',
        score REAL DEFAULT 0.0,
        total_actions INTEGER DEFAULT 0,
        current_level INTEGER DEFAULT 0,
        game_overs INTEGER DEFAULT 0,
        total_time REAL DEFAULT 0.0,
        level_stats JSONB,
        action_log JSONB DEFAULT '[]'::jsonb,
        started_at TIMESTAMPTZ DEFAULT NOW(),
        ended_at TIMESTAMPTZ,
        expires_at TIMESTAMPTZ NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_games_game_id ON games(game_id);
      CREATE INDEX IF NOT EXISTS idx_games_active ON games(is_active);
      CREATE INDEX IF NOT EXISTS idx_sessions_game ON play_sessions(game_id);
      CREATE INDEX IF NOT EXISTS idx_sessions_guid ON play_sessions(session_guid);
      CREATE INDEX IF NOT EXISTS idx_requests_status ON game_requests(status);
      CREATE INDEX IF NOT EXISTS idx_temp_guid ON temp_game_sessions(session_guid);

      CREATE TABLE IF NOT EXISTS app_logs (
        id TEXT PRIMARY KEY,
        level VARCHAR(10) NOT NULL DEFAULT 'info',
        source VARCHAR(100) NOT NULL DEFAULT 'app',
        message TEXT NOT NULL,
        metadata JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW()
      );

      CREATE INDEX IF NOT EXISTS idx_app_logs_created ON app_logs(created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_app_logs_level ON app_logs(level);
      CREATE INDEX IF NOT EXISTS idx_app_logs_source ON app_logs(source);

      CREATE TABLE IF NOT EXISTS app_settings (
        key VARCHAR(100) PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TIMESTAMPTZ DEFAULT NOW()
      );

      -- Default settings
      INSERT INTO app_settings (key, value) VALUES ('recording_enabled', 'true')
        ON CONFLICT (key) DO NOTHING;
    `);
    console.log(`[DB] PostgreSQL connected: ${process.env.DATABASE_URL?.replace(/\/\/.*@/, '//***@')}`);
  } finally {
    client.release();
  }
}

export default pool;

// ──── Helpers ────
export function genId(): string {
  return crypto.randomUUID();
}

/** Safely convert a JS value to a JSON string for JSONB columns. */
export function toJsonb(val: any): string | null {
  if (val === null || val === undefined) return null;
  if (typeof val === "string") return val; // already a string
  return JSON.stringify(val);
}

export async function queryOne(text: string, params?: any[]): Promise<any> {
  const result = await pool.query(text, params);
  return result.rows[0] || null;
}

export async function queryAll(text: string, params?: any[]): Promise<any[]> {
  const result = await pool.query(text, params);
  return result.rows;
}
