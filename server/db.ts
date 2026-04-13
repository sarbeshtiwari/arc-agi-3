import pg from "pg";

const { Pool } = pg;

const pool = new Pool({
  host: process.env.DB_HOST || "",
  port: parseInt(process.env.DB_PORT || "", 10),
  database: process.env.DB_NAME || "",
  user: process.env.DB_USER || "",
  password: process.env.DB_PASSWORD || "",
  max: 50,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000,
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
        display_name VARCHAR(100),
        is_admin BOOLEAN DEFAULT FALSE,
        is_active BOOLEAN DEFAULT TRUE,
        role VARCHAR(20) DEFAULT 'tasker',
        team_lead_id TEXT REFERENCES users(id) ON DELETE SET NULL,
        allowed_pages JSONB DEFAULT '[]'::jsonb,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
      );

      CREATE TABLE IF NOT EXISTS user_team_leads (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        lead_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(user_id, lead_id)
      );

      CREATE INDEX IF NOT EXISTS idx_utl_user ON user_team_leads(user_id);
      CREATE INDEX IF NOT EXISTS idx_utl_lead ON user_team_leads(lead_id);

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
        approval_status VARCHAR(20) DEFAULT 'draft',
        assigned_ql_id TEXT REFERENCES users(id) ON DELETE SET NULL,
        assigned_pl_id TEXT REFERENCES users(id) ON DELETE SET NULL,
        rejection_reason TEXT,
        rejection_by TEXT REFERENCES users(id) ON DELETE SET NULL,
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

      CREATE TABLE IF NOT EXISTS notifications (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        type VARCHAR(50) NOT NULL,
        title VARCHAR(255) NOT NULL,
        message TEXT NOT NULL DEFAULT '',
        game_id TEXT REFERENCES games(id) ON DELETE CASCADE,
        is_read BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMPTZ DEFAULT NOW()
      );

      CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id, is_read);
      CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications(created_at DESC);

      CREATE TABLE IF NOT EXISTS audit_log (
        id TEXT PRIMARY KEY,
        game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        action VARCHAR(50) NOT NULL,
        details JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW()
      );

      CREATE INDEX IF NOT EXISTS idx_audit_game ON audit_log(game_id, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);

      INSERT INTO app_settings (key, value) VALUES ('recording_enabled', 'true')
        ON CONFLICT (key) DO NOTHING;
    `);

    await migrateExistingData(client);

    console.log(`[DB] PostgreSQL connected: ${process.env.DB_HOST}:${process.env.DB_PORT}/${process.env.DB_NAME}`);
  } finally {
    client.release();
  }
}

async function migrateExistingData(client: pg.PoolClient) {
  const hasRoleCol = await client.query(
    `SELECT column_name FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'role'`
  );
  if (hasRoleCol.rows.length === 0) {
    await client.query(`ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) DEFAULT 'tasker'`);
    await client.query(`ALTER TABLE users ADD COLUMN IF NOT EXISTS team_lead_id TEXT REFERENCES users(id) ON DELETE SET NULL`);
    await client.query(`UPDATE users SET role = 'super_admin' WHERE is_admin = TRUE AND role = 'tasker'`);
    console.log('[DB] Migrated users: added role + team_lead_id columns');
  }

  const hasDisplayName = await client.query(
    `SELECT column_name FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'display_name'`
  );
  if (hasDisplayName.rows.length === 0) {
    await client.query(`ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR(100)`);
    console.log('[DB] Migrated users: added display_name column');
  }

  const hasJunction = await client.query(
    `SELECT table_name FROM information_schema.tables WHERE table_name = 'user_team_leads'`
  );
  if (hasJunction.rows.length === 0) {
    await client.query(`
      CREATE TABLE IF NOT EXISTS user_team_leads (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        lead_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(user_id, lead_id)
      );
      CREATE INDEX IF NOT EXISTS idx_utl_user ON user_team_leads(user_id);
      CREATE INDEX IF NOT EXISTS idx_utl_lead ON user_team_leads(lead_id);
    `);
    const legacyLeads = await client.query(
      `SELECT id, team_lead_id FROM users WHERE team_lead_id IS NOT NULL`
    );
    for (const row of legacyLeads.rows) {
      await client.query(
        `INSERT INTO user_team_leads (id, user_id, lead_id) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`,
        [crypto.randomUUID(), row.id, row.team_lead_id]
      );
    }
    if (legacyLeads.rows.length > 0) {
      console.log(`[DB] Migrated ${legacyLeads.rows.length} team_lead_id records to user_team_leads junction table`);
    }
  }

  const hasApprovalCol = await client.query(
    `SELECT column_name FROM information_schema.columns WHERE table_name = 'games' AND column_name = 'approval_status'`
  );
  if (hasApprovalCol.rows.length === 0) {
    await client.query(`ALTER TABLE games ADD COLUMN IF NOT EXISTS approval_status VARCHAR(20) DEFAULT 'draft'`);
    await client.query(`ALTER TABLE games ADD COLUMN IF NOT EXISTS assigned_ql_id TEXT REFERENCES users(id) ON DELETE SET NULL`);
    await client.query(`ALTER TABLE games ADD COLUMN IF NOT EXISTS assigned_pl_id TEXT REFERENCES users(id) ON DELETE SET NULL`);
    await client.query(`ALTER TABLE games ADD COLUMN IF NOT EXISTS rejection_reason TEXT`);
    await client.query(`ALTER TABLE games ADD COLUMN IF NOT EXISTS rejection_by TEXT REFERENCES users(id) ON DELETE SET NULL`);
    await client.query(`UPDATE games SET approval_status = 'approved' WHERE approval_status = 'draft'`);
    console.log('[DB] Migrated games: added approval columns, existing games set to approved');
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
