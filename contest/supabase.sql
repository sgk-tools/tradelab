-- tradelab: accounts and orders for the friends' contest (Supabase).
-- Run once in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- BEFORE running, replace 招待コードをここに書く (one place below) with your invite code.
-- Safe to run again: it keeps existing data and only (re)creates rules and functions.

create schema if not exists private;
revoke all on schema private from anon, authenticated;

create table if not exists private.settings (key text primary key, value text not null);
insert into private.settings (key, value) values ('invite_code', '招待コードをここに書く')
  on conflict (key) do update set value = excluded.value;

-- ---- profiles: one row per account, display name shown in the ranking -------------
create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  name text not null unique check (char_length(name) between 1 and 20),
  created_at timestamptz not null default now()
);
alter table public.profiles enable row level security;
drop policy if exists "profiles: signed-in users read" on public.profiles;
create policy "profiles: signed-in users read" on public.profiles
  for select to authenticated using (true);

-- sign-up needs the invite code (checked here, so the page cannot be bypassed)
create or replace function private.check_signup() returns trigger
language plpgsql security definer set search_path = '' as $$
declare code text;
begin
  select value into code from private.settings where key = 'invite_code';
  if coalesce(new.raw_user_meta_data ->> 'invite', '') <> coalesce(code, '') then
    raise exception 'invalid invite code';
  end if;
  new.raw_user_meta_data := new.raw_user_meta_data - 'invite';
  return new;
end $$;
drop trigger if exists tradelab_check_signup on auth.users;
create trigger tradelab_check_signup before insert on auth.users
  for each row execute function private.check_signup();

create or replace function private.create_profile() returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  insert into public.profiles (id, name)
  values (new.id, left(coalesce(nullif(trim(new.raw_user_meta_data ->> 'name'), ''),
                                split_part(new.email, '@', 1)), 20));
  return new;
end $$;
drop trigger if exists tradelab_create_profile on auth.users;
create trigger tradelab_create_profile after insert on auth.users
  for each row execute function private.create_profile();

-- ---- orders: users can only add their own; time and owner are set by the server ------
create table if not exists public.orders (
  id bigint generated always as identity primary key,
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  created_at timestamptz not null default now(),
  symbol text not null default '' check (char_length(symbol) <= 20),
  side text not null check (side in ('buy', 'sell', 'cancel')),
  type text not null default 'market' check (type in ('market', 'limit', 'stop')),
  qty double precision not null default 0 check (qty >= 0),
  px double precision, st double precision, tg double precision,
  target bigint,
  k text, cu text, fx double precision, asof date,
  auto boolean not null default false
);
create index if not exists orders_user_idx on public.orders (user_id, id);
alter table public.orders enable row level security;
drop policy if exists "orders: read own" on public.orders;
create policy "orders: read own" on public.orders
  for select to authenticated using (user_id = auth.uid());
drop policy if exists "orders: add own" on public.orders;
create policy "orders: add own" on public.orders
  for insert to authenticated with check (user_id = auth.uid());

create or replace function private.stamp_row() returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  new.user_id := auth.uid();
  if tg_table_name = 'orders' then new.created_at := now(); end if;
  if tg_table_name = 'exits' then new.set_at := now(); end if;
  return new;
end $$;
drop trigger if exists tradelab_stamp_order on public.orders;
create trigger tradelab_stamp_order before insert on public.orders
  for each row execute function private.stamp_row();

-- ---- exits: stop-loss / take-profit settings edited on a holding ----------------------
create table if not exists public.exits (
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  symbol text not null check (char_length(symbol) <= 20),
  k text,
  enabled boolean not null default true,
  st double precision, tg double precision,
  set_at timestamptz not null default now(),
  primary key (user_id, symbol)
);
alter table public.exits enable row level security;
drop policy if exists "exits: own" on public.exits;
create policy "exits: own" on public.exits
  for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());
drop trigger if exists tradelab_stamp_exit on public.exits;
create trigger tradelab_stamp_exit before insert or update on public.exits
  for each row execute function private.stamp_row();

-- ---- favourites -----------------------------------------------------------------------
create table if not exists public.favorites (
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  symbol text not null check (char_length(symbol) <= 20),
  added_at timestamptz not null default now(),
  primary key (user_id, symbol)
);
alter table public.favorites enable row level security;
drop policy if exists "favorites: own" on public.favorites;
create policy "favorites: own" on public.favorites
  for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());

grant select on public.profiles to authenticated;
grant select, insert on public.orders to authenticated;
grant select, insert, update, delete on public.exits, public.favorites to authenticated;
