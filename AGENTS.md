# 作業ルール（AI エージェント共通）

このファイルは Codex がそのまま読み、Claude Code は CLAUDE.md 経由で読み込む（内容は1か所だけ）。
ルールを変えるときはこのファイルだけを更新する。現在の状況と未決事項は HANDOFF.md にある。

## 1. このリポジトリ

- tradelab: CAN SLIM・ダウ理論・スイングトレード手法で銘柄を毎日判定し、ランキングを公開する研究用ソフト。
- 公開ページ: https://sgk-tools.github.io/tradelab/ （GitHub Actions が毎朝 07:13 JST ごろ作り直して配信する）
- 対象: 米国株（S&P 500）、日本株（東証プライムの内国株）、主要ETF（日米 約60本）、主要暗号資産12銘柄。投資信託は対象外。
- 発注機能は持たない。実資金は動かない。デモトレードはブラウザ内の仮想売買。

## 2. 依頼者との会話ルール（人間向けの文章すべて）

- 一人称「わたし」、二人称「きみ」。常体（〜だ、〜である）。敬語は使わない。
- 結論を先に述べる。返答の型は「結論／理由／注意点（要確認事項）／次にやること」。
- 事実・推測・未確認を区別する。断定できないことは「わからない」「要確認」と書く。
- 過剰な励まし、長い前置き、曖昧な断定はしない。
- 回答冒頭に、取得できる場合は現在の日本時間を書く。
- コード、JSON、テスト結果、コミットメッセージにはこの口調を適用しない。

## 3. 作業の原則（厳守）

- 正常に動いている部分を壊さない。直す前に原因を調べ、指示された範囲だけを変える。
- 実行・検証していないものを完成と報告しない。可能な限り自動テストで確かめる。
- 重大な変更、ファイル削除、外部への送信・公開は、事前に依頼者の承認を得る。
- 完了報告は「変更内容、検証結果、残存する問題、未確認事項」をそろえる。
- 投資助言はしない。出力は「条件に合致した銘柄の一覧」であり「推奨」ではない、という表現を保つ。

## 4. Git の運用（Claude と Codex の衝突を防ぐ）

- 作業はブランチで行う。命名: Claude は `claude/<内容>`、Codex は `codex/<内容>`。
- main にはプルリクエストで取り込む。依頼者の指示があれば直接取り込んでよい。
- 毎朝の自動更新が main に `state/` をコミットする。作業を始める前と取り込む前に、必ず最新の main を取り込む（`git fetch origin main` → `git rebase origin/main`）。
- 同じ時間に両方のエージェントが作業しない。始める前に HANDOFF.md の「作業中」欄を確かめ、自分の作業を書く。終わったら消す。
- `state/data.json`（前日の順位）と `state/ledger.json`（紙上売買の記録）は手で編集しない。消すと記録が失われる。
- `_site/`（生成されたページ）、`data/cache/` はコミットしない。

## 5. コマンド

```bash
pip install "pandas>=2.0" "numpy>=1.24" "matplotlib>=3.7" "requests>=2.28" "xlrd>=2.0" "openpyxl>=3.1" pytest
```

```bash
python -m pytest -q
```

```bash
python -m tradelab app --out _site --state state --watch watch.txt
```

- 一部の市場だけ作る: `--markets us,crypto`（us, jp, etf, crypto）
- ローカルで試すときは `--state` に state/ の写しを渡す。本物の state/ を書き換えないこと。
- 価格は Yahoo Finance（query1.finance.yahoo.com）、S&P 500 構成銘柄は raw.githubusercontent.com、東証の銘柄一覧は www.jpx.co.jp から取得する。接続できない環境では、その市場の生成は失敗する。

## 6. 構成

```
tradelab/
  data.py          価格・銘柄情報の取得（Yahoo Finance）、CSV読み込み、整合性検査
  fundamentals.py  決算データの取得
  screen.py        CAN SLIM + ダウ理論の判定、銘柄一覧（S&P 500 / 東証プライム / ETF / 暗号資産）
  swing.py         トレンドテンプレート、仕掛けシグナル（BO・PP・PB・VCP）、取引計画、過去成績
  ledger.py        紙上売買の自動記録（シグナル上位10銘柄を毎日記録し、価格から決済を再計算）
  contest.py       仲間内の大会（Supabase の口座の注文を日足で約定させ、週間・月間・年間・累計の利益率で順位付け。旧方式の Google フォームにも対応）
  app.py           ページ用データの組み立てと HTML 出力
  app_template.html ページ本体（ランキング・お気に入り・検索・デモトレード・確認画面）
  cli.py           コマンド（fetch / backtest / walkforward / screen / app）
  engine.py, strategies.py, metrics.py, validate.py, report.py, costs.py  バックテスト基盤
tests/             自動テスト（変更したら必ず全件実行）
state/             毎日の引き継ぎデータ（自動更新がコミットする）
watch.txt          お気に入りのうち一覧外の銘柄（1行1銘柄）
contest/           大会の設定（config.json）、Supabase の表と規則（supabase.sql）、準備手順（SUPABASE.md、旧方式の SETUP.md）
.github/workflows/daily.yml  毎朝の更新と配信
```

## 7. 判定ルールの要点（変える場合は依頼者の承認が必要）

- 質の条件: C・A・N・S・L・DOW・TT。ETF と暗号資産は N・S・L・DOW・TT のみ。
- 仕掛けシグナル: BO（ブレイクアウト）、PP（ポケットピボット）、PB（RSI2 押し目）、VCP（値幅収縮・簡易版）。
- 点数 = 質の条件の合格数 + シグナルがあれば1点。候補 = 質の条件すべて合格 + シグナルあり + 市場条件 M 合格。
- 取引計画（オニールの規則）: 損切り −7%、利確 +20%、保有は最長60営業日。
- 決算データが取れない C・A は「判定不能（－）」で、合格に数えない。
- デモトレード: 成行は次の取引日の始値。指値・逆指値は日足の高値・安値で判定し、取消まで有効（最長60取引日）。日本株は100株単位（設定で1株）。
