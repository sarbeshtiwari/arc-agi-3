import { queryAll } from "../db.js";

export async function getUserLeadIds(userId: string): Promise<string[]> {
  const rows = await queryAll(
    `SELECT lead_id FROM user_team_leads WHERE user_id = $1`,
    [userId],
  );
  return rows.map((r: any) => r.lead_id);
}

export async function getUserLeadUsernames(userId: string): Promise<string[]> {
  const rows = await queryAll(
    `SELECT u.username FROM user_team_leads utl
     JOIN users u ON u.id = utl.lead_id
     WHERE utl.user_id = $1
     ORDER BY u.username`,
    [userId],
  );
  return rows.map((r: any) => r.username);
}
