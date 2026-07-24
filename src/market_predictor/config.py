from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from market_predictor.app_config import config_get, load_app_config

load_dotenv()


class Settings(BaseSettings):
    alpaca_api_key_id: str | None = Field(default=None, alias="ALPACA_API_KEY_ID")
    alpaca_api_secret_key: str | None = Field(default=None, alias="ALPACA_API_SECRET_KEY")
    alpaca_stock_feed: str = Field(default="sip", alias="ALPACA_STOCK_FEED")
    alpaca_trading_base_url: str = Field(default="https://api.alpaca.markets", alias="ALPACA_TRADING_BASE_URL")
    app_config_path: Path = Field(default=Path("configs/default.toml"), alias="APP_CONFIG_PATH")
    sec_user_agent: str = Field(
        default="market-predictor/0.1 contact@example.com",
        alias="SEC_USER_AGENT",
    )
    reddit_client_id: str | None = Field(default=None, alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str | None = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    reddit_username: str | None = Field(default=None, alias="REDDIT_USERNAME")
    reddit_password: str | None = Field(default=None, alias="REDDIT_PASSWORD")
    reddit_user_agent: str = Field(
        default="market-predictor/0.1 by unknown",
        alias="REDDIT_USER_AGENT",
    )
    rapidapi_key: str | None = Field(default=None, alias="RAPIDAPI_KEY")
    finviz_elite_auth: str | None = Field(default=None, alias="FINVIZ_ELITE_AUTH")
    seeking_alpha_account_email: str | None = Field(default=None, alias="SEEKING_ALPHA_ACCOUNT_EMAIL")
    seeking_alpha_account_password: str | None = Field(default=None, alias="SEEKING_ALPHA_ACCOUNT_PASSWORD")
    seeking_alpha_access_token_cache_file: Path = Field(
        default=Path("data/cache/seeking_alpha/access_token.json"),
        alias="SEEKING_ALPHA_ACCESS_TOKEN_CACHE_FILE",
    )
    finbert_model: str = Field(default="ProsusAI/finbert", alias="FINBERT_MODEL")
    azure_storage_connection_string: str | None = Field(default=None, alias="AZURE_STORAGE_CONNECTION_STRING")
    azure_storage_account_url: str | None = Field(default=None, alias="AZURE_STORAGE_ACCOUNT_URL")
    azure_storage_container: str = Field(default="market-data", alias="AZURE_STORAGE_CONTAINER")
    azure_blob_prefix: str = Field(default="market-predictor", alias="AZURE_BLOB_PREFIX")
    runtime_memory_budget_gib: float = Field(default=4.0, alias="RUNTIME_MEMORY_BUDGET_GIB", gt=0)
    runtime_memory_headroom_gib: float = Field(default=0.25, alias="RUNTIME_MEMORY_HEADROOM_GIB", gt=0)
    runtime_max_concurrent_inference: int = Field(
        default=1,
        alias="RUNTIME_MAX_CONCURRENT_INFERENCE",
        ge=1,
    )
    runtime_max_tickers_per_request: int = Field(
        default=100,
        alias="RUNTIME_MAX_TICKERS_PER_REQUEST",
        ge=1,
    )
    runtime_inference_memory_reservation_gib: float = Field(
        default=0.5,
        alias="RUNTIME_INFERENCE_MEMORY_RESERVATION_GIB",
        gt=0,
    )
    runtime_reject_unknown_memory: bool = Field(
        default=False,
        alias="RUNTIME_REJECT_UNKNOWN_MEMORY",
    )
    api_environment: str = Field(default="production", alias="API_ENVIRONMENT")
    api_auth_mode: str = Field(default="entra", alias="API_AUTH_MODE")
    api_jwt_issuer: str | None = Field(default=None, alias="API_JWT_ISSUER")
    api_jwt_audience: str | None = Field(default=None, alias="API_JWT_AUDIENCE")
    api_jwks_path: Path | None = Field(default=None, alias="API_JWKS_PATH")
    api_development_bearer_token: SecretStr | None = Field(
        default=None,
        alias="API_DEVELOPMENT_BEARER_TOKEN",
    )
    api_maximum_body_bytes: int = Field(
        default=65_536,
        alias="API_MAXIMUM_BODY_BYTES",
        ge=1_024,
        le=1_048_576,
    )
    api_maximum_rate_limit_principals: int = Field(
        default=10_000,
        alias="API_MAXIMUM_RATE_LIMIT_PRINCIPALS",
        ge=1,
    )
    api_prediction_requests_per_minute: int = Field(
        default=60,
        alias="API_PREDICTION_REQUESTS_PER_MINUTE",
        ge=1,
    )
    api_operations_requests_per_minute: int = Field(
        default=30,
        alias="API_OPERATIONS_REQUESTS_PER_MINUTE",
        ge=1,
    )
    api_metrics_requests_per_minute: int = Field(
        default=30,
        alias="API_METRICS_REQUESTS_PER_MINUTE",
        ge=1,
    )
    api_replay_requests_per_minute: int = Field(
        default=5,
        alias="API_REPLAY_REQUESTS_PER_MINUTE",
        ge=1,
    )
    api_replay_enabled: bool = Field(default=False, alias="API_REPLAY_ENABLED")
    data_dir: Path = Path("data")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def has_alpaca(self) -> bool:
        return bool(self.alpaca_api_key_id and self.alpaca_api_secret_key)

    @property
    def has_azure_storage(self) -> bool:
        return bool(self.azure_storage_connection_string or self.azure_storage_account_url)

    @property
    def azure_prefix(self) -> str:
        return self.azure_blob_prefix.strip("/")

    @property
    def app_config(self) -> dict[str, Any]:
        return load_app_config(self.app_config_path)

    @property
    def has_reddit(self) -> bool:
        return bool(
            self.reddit_client_id
            and self.reddit_client_secret
            and self.reddit_username
            and self.reddit_password
        )

    @property
    def reddit_subreddit_list(self) -> list[str]:
        return list(
            config_get(
                self.app_config,
                "reddit.subreddits",
                ["stocks", "investing", "SecurityAnalysis", "wallstreetbets"],
            )
        )

    @property
    def reddit_limit_per_subreddit(self) -> int:
        return int(config_get(self.app_config, "reddit.limit_per_subreddit", 50))

    @property
    def reddit_search_time_filter(self) -> str:
        return str(config_get(self.app_config, "reddit.search_time_filter", "year"))

    @property
    def reddit_include_post_comments(self) -> bool:
        return bool(config_get(self.app_config, "reddit.include_post_comments", True))

    @property
    def reddit_comments_per_post(self) -> int:
        return int(config_get(self.app_config, "reddit.comments_per_post", 25))

    @property
    def reddit_ticker_false_positive_stoplist(self) -> set[str]:
        return set(config_get(self.app_config, "reddit.ticker_false_positive_stoplist", []))

    @property
    def universe_asset_class(self) -> str:
        return str(config_get(self.app_config, "universe.asset_class", "us_equity"))

    @property
    def universe_status(self) -> str:
        return str(config_get(self.app_config, "universe.status", "active"))

    @property
    def universe_exchanges(self) -> set[str]:
        return set(config_get(self.app_config, "universe.exchanges", ["NYSE", "NASDAQ", "AMEX"]))

    @property
    def universe_tradable_only(self) -> bool:
        return bool(config_get(self.app_config, "universe.tradable_only", True))

    @property
    def has_seeking_alpha_rapidapi(self) -> bool:
        return bool(self.rapidapi_key and self.seeking_alpha_rapidapi_host)

    @property
    def has_seeking_alpha_account_credentials(self) -> bool:
        return bool(self.seeking_alpha_account_email and self.seeking_alpha_account_password)

    @property
    def seeking_alpha_rapidapi_host(self) -> str:
        return str(config_get(self.app_config, "seeking_alpha.rapidapi_host", "seeking-alpha.p.rapidapi.com"))

    @property
    def seeking_alpha_analysis_endpoint(self) -> str:
        return str(config_get(self.app_config, "seeking_alpha.analysis_endpoint", "/analysis/v2/list"))

    @property
    def seeking_alpha_analysis_params(self) -> str:
        return str(config_get(self.app_config, "seeking_alpha.analysis_params", "id={ticker_lower}&size=40&number=1"))

    @property
    def seeking_alpha_ratings_endpoint(self) -> str:
        return str(config_get(self.app_config, "seeking_alpha.ratings_endpoint", "/symbols/get-ratings"))

    @property
    def seeking_alpha_ratings_params(self) -> str:
        return str(config_get(self.app_config, "seeking_alpha.ratings_params", "symbol={ticker}"))

    @property
    def swing_seed_tickers(self) -> list[str]:
        return [str(item).upper() for item in config_get(self.app_config, "swing_universe.seed_tickers", [])]

    @property
    def swing_candidate_tickers(self) -> list[str]:
        values = [str(item).upper() for item in config_get(self.app_config, "swing_universe.candidates", [])]
        return list(dict.fromkeys(values))

    @property
    def market_benchmark_ticker(self) -> str:
        return str(config_get(self.app_config, "market_benchmarks.market", "SPY")).upper()

    @property
    def sector_benchmarks(self) -> dict[str, str]:
        return {
            str(sector): str(ticker).upper()
            for sector, ticker in dict(config_get(self.app_config, "sector_benchmarks", {})).items()
        }

    @property
    def sector_ticker_groups(self) -> dict[str, list[str]]:
        groups = dict(config_get(self.app_config, "sector_tickers", {}))
        return {
            str(sector): [str(ticker).upper() for ticker in tickers]
            for sector, tickers in groups.items()
        }

    @property
    def ticker_sector_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for sector, tickers in self.sector_ticker_groups.items():
            for ticker in tickers:
                mapping.setdefault(ticker.upper(), sector)
        return mapping

    def sector_for_ticker(self, ticker: str) -> str | None:
        return self.ticker_sector_map.get(ticker.upper())

    def sector_benchmark_for_ticker(self, ticker: str) -> str:
        sector = self.sector_for_ticker(ticker)
        if sector and sector in self.sector_benchmarks:
            return self.sector_benchmarks[sector]
        return self.market_benchmark_ticker

    @property
    def max_workers(self) -> int:
        return max(1, int(config_get(self.app_config, "performance.max_workers", 6)))

    @property
    def finbert_batch_size(self) -> int:
        return max(1, int(config_get(self.app_config, "performance.finbert_batch_size", 32)))

    @property
    def torch_num_threads(self) -> int:
        return max(0, int(config_get(self.app_config, "performance.torch_num_threads", 0)))

    @property
    def seeking_alpha_monthly_request_limit(self) -> int:
        return int(config_get(self.app_config, "seeking_alpha.monthly_request_limit", 200))

    @property
    def seeking_alpha_analysis_cache_hours(self) -> int:
        return int(config_get(self.app_config, "seeking_alpha.analysis_cache_hours", 24))

    @property
    def seeking_alpha_ratings_cache_hours(self) -> int:
        return int(config_get(self.app_config, "seeking_alpha.ratings_cache_hours", 24))

    @property
    def seeking_alpha_usage_file(self) -> Path:
        return Path(str(config_get(self.app_config, "seeking_alpha.usage_file", "data/usage/rapidapi_usage.json")))

    @property
    def seeking_alpha_cache_dir(self) -> Path:
        return Path(str(config_get(self.app_config, "seeking_alpha.cache_dir", "data/cache/seeking_alpha")))

    @property
    def seeking_alpha_fail_when_monthly_limit_reached(self) -> bool:
        return bool(config_get(self.app_config, "seeking_alpha.fail_when_monthly_limit_reached", True))

    @property
    def seeking_alpha_access_token_endpoint(self) -> str:
        return str(config_get(self.app_config, "seeking_alpha.access_token_endpoint", "/accounts/get-access-token"))

    @property
    def seeking_alpha_access_token_cache_hours(self) -> int:
        return int(config_get(self.app_config, "seeking_alpha.access_token_cache_hours", 12))

    @property
    def seeking_alpha_event_feeds(self) -> list[dict[str, Any]]:
        return list(
            config_get(
                self.app_config,
                "seeking_alpha.event_feeds",
                [
                    {
                        "name": "analysis",
                        "endpoint": self.seeking_alpha_analysis_endpoint,
                        "params": self.seeking_alpha_analysis_params,
                        "cache_hours": self.seeking_alpha_analysis_cache_hours,
                        "limit": 40,
                    }
                ],
            )
        )

    @property
    def seeking_alpha_snapshot_feeds(self) -> list[dict[str, Any]]:
        return list(
            config_get(
                self.app_config,
                "seeking_alpha.snapshot_feeds",
                [
                    {
                        "name": "ratings",
                        "endpoint": self.seeking_alpha_ratings_endpoint,
                        "params": self.seeking_alpha_ratings_params,
                        "cache_hours": self.seeking_alpha_ratings_cache_hours,
                    }
                ],
            )
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
