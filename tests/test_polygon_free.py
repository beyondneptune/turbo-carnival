"""Mocked tests for ingest.polygon_free; no real network calls."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ingest.polygon_free import (
    BASE_URL,
    FREE_TIER_RPM,
    PolygonAuthError,
    PolygonError,
    PolygonFreeClient,
    PolygonRateLimitError,
    get_api_key,
)


def _mock_response(json_data, status_code: int = 200, text: str = "") -> MagicMock:
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_data
    m.text = text or str(json_data)
    return m


def _client(session: MagicMock | None = None, **kwargs) -> PolygonFreeClient:
    return PolygonFreeClient(api_key="test_key", session=session or MagicMock(), **kwargs)


# ---------------- get_api_key ----------------


def test_get_api_key_returns_env_value(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test_key_abc")
    assert get_api_key() == "test_key_abc"


def test_get_api_key_returns_none_when_unset(monkeypatch, tmp_path, chdir_tmp_root):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    # Ensure no .env in CWD or repo root would clobber this. python-dotenv loads
    # the path we pass; we point it at an empty tmp_path with no .env file.
    assert get_api_key() is None


@pytest.fixture
def chdir_tmp_root(tmp_path, monkeypatch):
    """Run with cwd = tmp_path so any auto .env discovery misses the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------- PolygonFreeClient init ----------------


def test_client_init_raises_when_no_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    # Avoid load_dotenv pulling a real key from the repo's .env if it exists.
    monkeypatch.setattr(
        "ingest.polygon_free.get_api_key", lambda: None
    )
    with pytest.raises(PolygonAuthError):
        PolygonFreeClient()


def test_client_accepts_explicit_api_key():
    client = PolygonFreeClient(api_key="x", session=MagicMock())
    assert client is not None


def test_client_default_base_url():
    client = _client()
    assert BASE_URL.startswith("https://api.polygon.io")
    # rstrip should not introduce/strip anything for default
    assert client._base_url == BASE_URL.rstrip("/")


def test_client_default_rate_limit_matches_free_tier():
    client = _client()
    expected_interval = 60.0 / FREE_TIER_RPM
    assert client._min_interval_s == expected_interval


def test_client_invalid_rate_limit_raises():
    with pytest.raises(ValueError):
        PolygonFreeClient(api_key="x", session=MagicMock(), rate_limit_per_min=0)


# ---------------- _get error mapping ----------------


def test_get_raises_rate_limit_on_429():
    session = MagicMock()
    session.get.return_value = _mock_response({"error": "rate limited"}, status_code=429)
    client = _client(session=session, rate_limit_per_min=600)
    with pytest.raises(PolygonRateLimitError):
        client._get("/v2/aggs/ticker/SPY/range/1/day/2026-01-01/2026-01-02")


def test_get_raises_auth_on_401():
    session = MagicMock()
    session.get.return_value = _mock_response({"error": "unauthorized"}, status_code=401)
    client = _client(session=session, rate_limit_per_min=600)
    with pytest.raises(PolygonAuthError):
        client._get("/whatever")


def test_get_raises_auth_on_403():
    session = MagicMock()
    session.get.return_value = _mock_response({"error": "forbidden"}, status_code=403)
    client = _client(session=session, rate_limit_per_min=600)
    with pytest.raises(PolygonAuthError):
        client._get("/whatever")


def test_get_raises_polygon_error_on_500():
    session = MagicMock()
    session.get.return_value = _mock_response({"error": "boom"}, status_code=500)
    client = _client(session=session, rate_limit_per_min=600)
    with pytest.raises(PolygonError):
        client._get("/whatever")


def test_get_passes_api_key_in_params():
    session = MagicMock()
    session.get.return_value = _mock_response({"results": []})
    client = _client(session=session, rate_limit_per_min=600)
    client._get("/v2/aggs/ticker/SPY/range/1/day/2026-01-01/2026-01-02")
    _, kwargs = session.get.call_args
    assert kwargs["params"]["apiKey"] == "test_key"


# ---------------- fetch_aggregates ----------------


def test_fetch_aggregates_happy(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test_key")
    session = MagicMock()
    session.get.return_value = _mock_response(
        {
            "status": "OK",
            "results": [
                {
                    "t": 1700000000000,
                    "o": 1.0,
                    "h": 2.0,
                    "l": 0.5,
                    "c": 1.5,
                    "v": 100,
                    "vw": 1.4,
                    "n": 5,
                }
            ],
        }
    )
    client = PolygonFreeClient(session=session, rate_limit_per_min=600)
    df = client.fetch_aggregates("SPY", "2026-01-01", "2026-01-31")
    assert len(df) == 1
    assert list(df.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
    ]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"
    row = df.iloc[0]
    assert row["open"] == 1.0
    assert row["close"] == 1.5
    assert row["transactions"] == 5


def test_fetch_aggregates_empty_results():
    session = MagicMock()
    session.get.return_value = _mock_response({"status": "OK", "results": []})
    client = _client(session=session, rate_limit_per_min=600)
    df = client.fetch_aggregates("SPY", "2026-01-01", "2026-01-31")
    assert df.empty
    assert list(df.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
    ]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"


def test_fetch_aggregates_missing_results_key():
    session = MagicMock()
    session.get.return_value = _mock_response({"status": "OK"})
    client = _client(session=session, rate_limit_per_min=600)
    df = client.fetch_aggregates("SPY", "2026-01-01", "2026-01-31")
    assert df.empty


def test_fetch_aggregates_url_includes_path_segments():
    session = MagicMock()
    session.get.return_value = _mock_response({"status": "OK", "results": []})
    client = _client(session=session, rate_limit_per_min=600)
    client.fetch_aggregates("AAPL", "2026-01-01", "2026-01-31")
    args, _ = session.get.call_args
    url = args[0]
    assert "/v2/aggs/ticker/AAPL/range/1/day/2026-01-01/2026-01-31" in url


# ---------------- fetch_options_contracts ----------------


def test_fetch_options_contracts_happy():
    session = MagicMock()
    session.get.return_value = _mock_response(
        {
            "status": "OK",
            "results": [
                {
                    "ticker": "O:SPY260117C00600000",
                    "underlying_ticker": "SPY",
                    "contract_type": "call",
                    "strike_price": 600.0,
                    "expiration_date": "2026-01-17",
                    "exercise_style": "american",
                    "shares_per_contract": 100,
                },
                {
                    "ticker": "O:SPY260117P00600000",
                    "underlying_ticker": "SPY",
                    "contract_type": "put",
                    "strike_price": 600.0,
                    "expiration_date": "2026-01-17",
                    "exercise_style": "american",
                    "shares_per_contract": 100,
                },
            ],
        }
    )
    client = _client(session=session, rate_limit_per_min=600)
    df = client.fetch_options_contracts("SPY")
    assert len(df) == 2
    assert set(df.columns) == {
        "ticker",
        "underlying_ticker",
        "contract_type",
        "strike_price",
        "expiration_date",
        "exercise_style",
        "shares_per_contract",
    }
    assert df.iloc[0]["contract_type"] == "call"
    assert df.iloc[1]["contract_type"] == "put"


def test_fetch_options_contracts_empty():
    session = MagicMock()
    session.get.return_value = _mock_response({"status": "OK", "results": []})
    client = _client(session=session, rate_limit_per_min=600)
    df = client.fetch_options_contracts("SPY")
    assert df.empty
    assert list(df.columns) == [
        "ticker",
        "underlying_ticker",
        "contract_type",
        "strike_price",
        "expiration_date",
        "exercise_style",
        "shares_per_contract",
    ]


def test_fetch_options_contracts_passes_filter_params():
    session = MagicMock()
    session.get.return_value = _mock_response({"status": "OK", "results": []})
    client = _client(session=session, rate_limit_per_min=600)
    client.fetch_options_contracts(
        "SPY", expiration_date="2026-01-17", contract_type="call", limit=50
    )
    _, kwargs = session.get.call_args
    params = kwargs["params"]
    assert params["underlying_ticker"] == "SPY"
    assert params["expiration_date"] == "2026-01-17"
    assert params["contract_type"] == "call"
    assert params["limit"] == 50
    # None-valued filters should be dropped
    assert "as_of" not in params


# ---------------- fetch_options_aggregates ----------------


def test_fetch_options_aggregates_uses_option_ticker_in_url():
    session = MagicMock()
    session.get.return_value = _mock_response({"status": "OK", "results": []})
    client = _client(session=session, rate_limit_per_min=600)
    option_ticker = "O:SPY260117C00600000"
    client.fetch_options_aggregates(option_ticker, "2026-01-01", "2026-01-17")
    args, _ = session.get.call_args
    url = args[0]
    assert option_ticker in url
    assert "/v2/aggs/ticker/" in url


# ---------------- throttling ----------------


def test_throttle_enforces_min_interval():
    session = MagicMock()
    session.get.return_value = _mock_response({"status": "OK", "results": []})
    # 600 rpm => 0.1s between calls
    client = _client(session=session, rate_limit_per_min=600)
    start = time.monotonic()
    client._get("/x")
    client._get("/x")
    elapsed = time.monotonic() - start
    assert elapsed >= 0.1


def test_throttle_does_not_oversleep_when_idle():
    session = MagicMock()
    session.get.return_value = _mock_response({"status": "OK", "results": []})
    client = _client(session=session, rate_limit_per_min=600)
    # First call should not block since _last_request_at = 0.0 and >>0.1s have
    # already elapsed since program start.
    start = time.monotonic()
    client._get("/x")
    elapsed = time.monotonic() - start
    assert elapsed < 0.5
