# tradelab

売買戦略を過去データで検証するための研究用ソフト。株・FX・仮想通貨を同じ仕組みで扱う。
発注機能は持たない。実資金は動かない。

## 公開ページと毎朝の自動更新（GitHub Pages）

- 公開ページ: `docs/index.html`（GitHub Pages で `main` ブランチの `/docs` を公開する）。
- 毎朝 07:13 JST ごろ、GitHub Actions（`.github/workflows/daily.yml`）が次を行う: テスト → 価格取得とランキング作成（`python -m tradelab app --out docs --watch watch.txt`）→ `docs/` をコミット。前日の `docs/data.json`（順位変動）と `docs/ledger.json`（紙上売買の記録）を引き継ぐ。
- 手動で動かすときは、GitHub の Actions タブ → daily-update → Run workflow。
- お気に入りに S&P 500・主要暗号資産以外の銘柄を入れるときは、`watch.txt` に Yahoo Finance の銘柄コードを書く（例: `7203.T`）。
- 公開ページのデモトレードとお気に入りは、各自のブラウザにだけ保存される（端末間で共有されない）。
- 条件に合致した銘柄の一覧であり、売買の推奨ではない。データは Yahoo Finance 由来で、誤りや遅延を含みうる。

### 初回の設定

1. このリポジトリを公開（Public）にする。
2. Settings → Pages → Build and deployment: Source を「Deploy from a branch」、Branch を `main` / `/docs` にして保存。
3. Settings → Actions → General → Workflow permissions を「Read and write permissions」にする。
4. Actions タブで daily-update を一度 Run workflow して、成功を確かめる。

## できること

- 価格データ（CSV または Yahoo Finance からの取得）の読み込みと整合性検査
- 手数料・スプレッド・最小売買単位・空売り可否を織り込んだバックテスト
- ウォークフォワード検証（パラメータを選んだ期間とは別の期間で成績を測る）
- 買い持ち（buy and hold）との比較、HTML レポート出力

## 使い方

```bash
pip install pandas numpy matplotlib requests pytest
```

```bash
python -m tradelab fetch --symbol BTC-USD --start 2015-01-01 --out data/btcusd.csv
```

```bash
python -m tradelab backtest --csv data/btcusd.csv --market crypto --strategy sma_cross --params fast=20 slow=100 --capital 1876 --out reports/bt.html
```

```bash
python -m tradelab walkforward --csv data/btcusd.csv --market crypto --strategy donchian_breakout --grid entry=20,55,100 exit=10,20,50 --train 750 --test 250 --capital 1876 --out reports/wf.html
```

```bash
python examples/scan.py
```

```bash
python -m pytest -q
```

Yahoo のシンボル例: `7203.T`（国内株）, `SPY`（米国株）, `USDJPY=X`（FX）, `BTC-USD`（仮想通貨）。

## 銘柄スクリーニング（CAN SLIM + ダウ理論）

条件に合致する銘柄を一覧にする。発注はしない。結果は HTML と CSV に出力する。

```bash
python -m tradelab screen --universe sp500 --out reports/screen_sp500.html
```

```bash
python -m tradelab screen --file my_symbols.txt --set c_min_eps_growth=0.2 dow_swing=10 --out reports/screen_my.html
```

```bash
python -m tradelab screen --universe crypto --out reports/screen_crypto.html
```

| 条件 | 内容（既定値） |
|---|---|
| C | 直近四半期の希薄化後EPSが前年同期比 +25% 以上 |
| A | 年次EPSの年平均成長率 +25% 以上、毎年増益、ROE 17% 以上 |
| N | 終値が52週高値から15%以内 |
| S | 直近50日の上昇日出来高 ÷ 下落日出来高が1.1以上 |
| L | 相対的強さが対象銘柄内の上位20%以内 |
| I | 機関投資家の保有動向。データを取得できないため未評価 |
| M | 市場指数（S&P 500）がダウ理論の上昇トレンド、かつ200日移動平均より上 |
| DOW | 銘柄自身がダウ理論の上昇トレンド（高値・安値の切り上げ） |

- 「候補」は C・A・N・S・L・DOW をすべて満たし、かつ M が合格の銘柄。M が不合格なら候補は0件になる。
- しきい値は `--set` で変更できる（項目名は `tradelab/screen.py` の `ScreenConfig`）。
- `--file` は1行1銘柄のテキスト。Yahoo Finance のシンボルで書く（例: `NVDA`, `BRK-B`）。
- `--universe crypto` は決算がないため N・S・L・DOW の4条件だけで判定する。CAN SLIM ではない。
- 取得データは `data/cache/<日付>/` に保存し、同じ日の再実行では再取得しない。
- 決算データは直近5四半期・4年分しか取得できない。このためスクリーニング規則そのものの過去検証はできていない。
- ダウ理論の売買規則だけは `--strategy dow_theory` でバックテストできる。

## スマートフォン用ランキングアプリ

米国株（S&P 500）と暗号資産を判定し、ランキング・株価チャート・決算推移を1つの HTML にまとめる。

```bash
python -m tradelab app --out dist
```

- `dist/index.html`: 単体で開けるページ。静的ホスティングに置けば iPhone のホーム画面に追加できる。
- `dist/page.html`: ページ本体のみ（公開ツールへの埋め込み用）。
- `dist/data.json`: その日のデータ。次回実行時に読み込み、前回からの順位変動を表示する。
- 評価不能な銘柄が20%を超えた場合は、不完全なランキングを出さずにエラーで終了する。

## スイングトレードの判定と紙上売買（v0.4.0）

`python -m tradelab app` のランキングに、次の項目を加えた（`tradelab/swing.py`, `tradelab/ledger.py`）。

| 区分 | 記号 | 内容 |
|---|---|---|
| 質の条件 | TT | トレンドテンプレート（ミネルヴィニ）。移動平均の並び、52週高安値の位置 |
| 仕掛けシグナル | BO | 直近20日高値の更新、出来高が50日平均の1.5倍以上 |
| 仕掛けシグナル | PP | ポケットピボット。上昇日の出来高が直近10日の下落日出来高を上回る |
| 仕掛けシグナル | PB | RSI(2) が10未満、200日平均より上（コナーズの押し目） |
| 仕掛けシグナル | VCP | 値幅の収縮（簡易版）、52週高値から15%以内 |

- 点数 = 質の条件（C A N S L DOW TT）の合格数 + シグナルが1つ以上なら1点。候補は全条件・シグナルあり・M 合格。
- 取引計画（オニールの売買規則）: 損切り = 終値 −7%、利確 = 終値 +20%、60営業日で手仕舞い。数量は画面で入れた資金と1回の損失上限から計算する。
- 過去成績: 同じ計画を過去データで再生し、シグナルごとの勝率と平均損益（R）を「どの日でも入る」と比べて表示する。生存者バイアスを含む。
- 紙上売買: 毎日、シグナルの出た上位10銘柄を自動で記録し、以後の価格で決済を判定する。記録は `dist/ledger.json`。毎回価格から再計算するため、実行を休んだ日があっても決済は漏れない。

## デモトレード（v0.5.0）

アプリ（公開ページ）上で仮想資金30万円の売買ができる。記録はページのクラウド保存に置き、所有者だけが読み書きできる。

- 注文種別: 成行（次の取引日の始値）、指値（買いは安値が指値以下、売りは高値が指値以上になった日。始値が有利なら始値）、逆指値（買いは高値が指定価格以上、売りは安値が指定価格以下になった日）。指値・逆指値は取消まで有効、最長60取引日。
- 確認画面: 確定前に、想定約定価格、約定代金、手数料、現金と買付余力の変化、保有数量、平均取得単価、資産に占める割合、損切り・利確で売れた場合の損益を表示する。
- 自動売却: 買い注文に損切り・利確の価格を付けると、約定後にその価格で全数量を売る。
- お気に入り: 最大50銘柄。検索して ☆ で追加する。S&P 500・主要暗号資産以外の銘柄（例: 7203.T）も追加でき、毎朝の更新で `--watch` ファイルとして取得・判定する（値動きの条件のみ、L は S&P 500 内の順位）。
- 円建て銘柄は円で、それ以外は取得した為替レートで円換算する。

## 前提と制約

- 約定タイミング: バー t の終値で判断し、バー t+1 の始値で約定する（先読み防止）。
- `--capital` は銘柄の建値通貨で指定する。米ドル建て銘柄なら円資金をドルに換算した額を渡す。為替変動は計算に含めない。
- `--market` のコスト設定（`tradelab/costs.py`）は一般的な水準を置いた仮定値。利用する業者の実際の手数料に置き換えること。
- FX のスワップポイント、株の配当課税、売買益への課税は計算に含めていない。
- 株価は分割・配当調整後の値を使う。過去の実際の株価とは異なるため、過去時点で最小売買単位を買えたかどうかの判定は正確ではない。
- 単一銘柄・日足を対象とする。複数銘柄のポートフォリオは未対応。
- 過去の成績は将来の成績を保証しない。

## 戦略の追加

`tradelab/strategies.py` に関数を追加し `STRATEGIES` に登録する。値は目標ポジション
（+1 買い、0 なし、-1 売り）。`tests/test_strategies.py` の `PARAMS` にも追加すると、
先読みをしていないかの検査が自動で適用される。
