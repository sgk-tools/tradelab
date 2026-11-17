# 大会の口座（ID・パスワード）の準備手順（Supabase）

所要 20〜30 分。Supabase の無料プランを使う（10 人程度なら無料枠に収まる）。
終わったら「Project URL」と「公開用の鍵（anon / publishable）」を Claude（または Codex）に渡す。
管理用の鍵（service_role / secret）は **チャットに貼らず**、GitHub の Secrets に自分で登録する。

## 1. プロジェクトを作る

1. https://supabase.com →「Start your project」→ GitHub でログイン。
2.「New project」: Name `tradelab`、Database Password は「Generate a password」で作り、パスワード管理アプリなどに保存する、Region は「Northeast Asia (Tokyo)」。→「Create new project」。数分待つ。

## 2. メール確認をオフにする（ID だけで登録できるようにする）

1. 左メニュー「Authentication」→「Sign In / Providers」（または「Providers」→「Email」）。
2.「Confirm email」をオフ、「Allow new users to sign up」をオンにして保存。

## 3. 表と不正防止の設定を入れる

1. 左メニュー「SQL Editor」→「New query」。
2. リポジトリの `contest/supabase.sql` の中身をすべて貼る。
3. 1 か所だけ書き換える: `'招待コードをここに書く'` を、友達に配る招待コードに変える（例: `'tl-2026-xxxx'`。引用符は残す）。
4.「Run」。「Success. No rows returned」と出れば完了。
- 招待コードを変えたいときは、同じ SQL のその行だけ書き換えて、もう一度 Run すればよい。

## 4. 鍵を2つ取り出す

左メニュー「Project Settings」→「API Keys」（または「API」）。

- **Project URL**（`https://xxxx.supabase.co`）と **公開用の鍵**（`anon` `public`、または `sb_publishable_…`）→ Claude に送ってよい（ページに埋め込む公開用の値）。
- **管理用の鍵**（`service_role` `secret`、または `sb_secret_…`）→ 次の手順で GitHub にだけ登録する。**チャットやメールに貼らない。**

## 5. 管理用の鍵を GitHub に登録する

1. https://github.com/sgk-tools/tradelab/settings/secrets/actions →「New repository secret」。
2. Name: `SUPABASE_SERVICE_KEY`、Secret: 管理用の鍵を貼る →「Add secret」。

## 6. 仕組み

- 画面からできるのは「自分の注文を追加する」「自分のお気に入り・自動売却の設定を変える」ことだけ。受付時刻と口座の持ち主はデータベース側で付ける。
- 約定と損益は毎朝の GitHub Actions が、管理用の鍵で全員の注文を読んで計算する。ランキングはその結果で、画面の見込みとは数円〜数十円ずれることがある。
- ID はログイン用（半角英数字）、表示名はランキングに出る名前。内部では `ID@sgk-tools.github.io` という形のメール欄に入るが、メールは送られない。
- パスワードを忘れた人は、主催者が Supabase の「Authentication」→「Users」でその人を削除し、登録し直してもらう（注文の記録も消える）。
