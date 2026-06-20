-- Optional: run in Supabase SQL editor if you want cloud history backup.
-- Local JSON history works without this table.

create table if not exists public.predictions (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  disease text not null,
  confidence double precision not null default 0,
  image_name text,
  display_name text,
  plant_type text,
  created_at timestamptz not null default now()
);

create index if not exists predictions_user_id_idx on public.predictions (user_id, created_at desc);

alter table public.predictions enable row level security;

-- Allow service role / anon key used by backend (adjust for your security model)
create policy "Allow backend insert predictions"
  on public.predictions for insert
  with check (true);

create policy "Allow backend select predictions"
  on public.predictions for select
  using (true);
