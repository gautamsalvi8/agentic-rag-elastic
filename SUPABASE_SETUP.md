# Supabase setup – login persists after app restart

Sessions are stored in Supabase so that when the app restarts (e.g. Streamlit Cloud sleep), users stay logged in.

## 1. Create free Supabase project

1. Go to [supabase.com](https://supabase.com) → Sign up / Login.
2. **New project** → choose org, name, DB password, region → Create.
3. Wait for the project to be ready.

## 2. Create the sessions table

1. In the project, open **SQL Editor**.
2. Copy the contents of **`supabase_sessions_table.sql`** from this repo.
3. Paste in the editor and click **Run**.

## 3. Get URL and key

1. Go to **Project Settings** (gear) → **API**.
2. Copy:
   - **Project URL** → use as `SUPABASE_URL`
   - **anon public** key → use as `SUPABASE_KEY` (safe for frontend; RLS is on)

## 4. Add to your app

**Local (.env):**
```env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Streamlit Cloud:**  
App → **Settings** → **Secrets** → add:
```toml
SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

Redeploy. After this, login will persist across app restarts.
