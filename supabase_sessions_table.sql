-- Run this in Supabase Dashboard → SQL Editor (one time)
-- Sessions persist here so login survives app restart (Streamlit Cloud / any host)

CREATE TABLE IF NOT EXISTS app_sessions (
  token TEXT PRIMARY KEY,
  username TEXT NOT NULL DEFAULT 'User',
  user_db_key TEXT NOT NULL DEFAULT '',
  auth_provider TEXT NOT NULL DEFAULT 'google',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Allow app (anon key) to insert, select, delete sessions
ALTER TABLE app_sessions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for app_sessions" ON app_sessions;
CREATE POLICY "Allow all for app_sessions" ON app_sessions FOR ALL USING (true) WITH CHECK (true);
