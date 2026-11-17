from tradelab.cli import main
from tradelab.data import synthetic_ohlcv


def test_backtest_command_writes_report(tmp_path, capsys):
    csv = tmp_path / "d.csv"
    synthetic_ohlcv(600, seed=1).to_csv(csv)
    out = tmp_path / "r.html"
    code = main(["backtest", "--csv", str(csv), "--market", "crypto", "--strategy",
                 "sma_cross", "--params", "fast=10", "slow=50", "--out", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "data:image/png;base64," in text and "シャープレシオ" in text
    assert "合成データ" not in text
    assert "buy_and_hold" in capsys.readouterr().out


def test_walkforward_command_on_synthetic(tmp_path, capsys):
    out = tmp_path / "wf.html"
    code = main(["walkforward", "--grid", "fast=5,10", "slow=30,60", "--train", "300",
                 "--test", "100", "--out", str(out)])
    assert code == 0 and "合成データ" in out.read_text(encoding="utf-8")
    assert "combinations tried: 4" in capsys.readouterr().out


def test_bad_input_returns_error_code(tmp_path, capsys):
    assert main(["backtest", "--csv", str(tmp_path / "missing.csv")]) == 2
    assert main(["backtest", "--params", "fast=50", "slow=10"]) == 2
    assert "error:" in capsys.readouterr().err
