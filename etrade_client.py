"""E*TRADE OAuth + trading client (Equities only).

This module is intentionally isolated from the signal engines so we can
enable/disable auto-exec without touching existing alert logic.

We use the official E*TRADE REST API (OAuth 1.0a). The flow is:
  1) get request token
  2) user authorizes -> verifier
  3) exchange for access token

Order workflow per E*TRADE docs:
  - Preview Order: POST /v1/accounts/{accountIdKey}/orders/preview
  - Place Order:   POST /v1/accounts/{accountIdKey}/orders/place
  - List Orders:   GET  /v1/accounts/{accountIdKey}/orders
  - Cancel Order:  PUT  /v1/accounts/{accountIdKey}/orders/cancel

Docs (sandbox base): https://apisb.etrade.com/docs/api/order/api-order-v1.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


try:
    # Prefer requests-oauthlib when available.
    from requests_oauthlib import OAuth1Session  # type: ignore

    _HAS_OAUTH = True
except Exception:
    OAuth1Session = None  # type: ignore
    _HAS_OAUTH = False


SANDBOX_BASE = "https://apisb.etrade.com"
LIVE_BASE = "https://api.etrade.com"


@dataclass
class OAuthTokens:
    oauth_token: str
    oauth_token_secret: str


class ETradeClient:
    """Thin E*TRADE REST client.

    Notes:
      - Equities only for now.
      - Bracket/OCO is NOT supported by E*TRADE API, so stop/TP are managed by our app.
    """

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        sandbox: bool = True,
        access_token: Optional[str] = None,
        access_token_secret: Optional[str] = None,
    ) -> None:
        if not _HAS_OAUTH:
            raise RuntimeError(
                "requests-oauthlib is required for E*TRADE integration. "
                "Add 'requests-oauthlib' to requirements.txt"
            )

        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.base = SANDBOX_BASE if sandbox else LIVE_BASE

        self._session = OAuth1Session(
            client_key=consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=access_token,
            resource_owner_secret=access_token_secret,
            signature_method="HMAC-SHA1",
        )

    # -------------------------
    # OAuth flow
    # -------------------------
    def get_request_token(self, callback_url: str = "oob") -> OAuthTokens:
        """Start OAuth by requesting a temporary request token.

        E*TRADE requires `oauth_callback` to be present in the signed OAuth
        parameter set for the request-token step. Some servers will reject
        requests where the callback is only passed as a plain query parameter.

        Using a dedicated OAuth1Session with `callback_uri` ensures the
        callback is embedded in the Authorization header (and therefore signed).
        """
        url = f"{self.base}/oauth/request_token"

        # Use a fresh session for the request-token step so callback_uri is
        # definitely included in the OAuth header.
        tmp = OAuth1Session(
            client_key=self.consumer_key,
            client_secret=self.consumer_secret,
            callback_uri=callback_url,
            signature_method="HMAC-SHA1",
        )
        resp = tmp.fetch_request_token(url)
        return OAuthTokens(resp["oauth_token"], resp["oauth_token_secret"])  # type: ignore[index]

    def get_authorize_url(self, request_token: str) -> str:
        return f"https://us.etrade.com/e/t/etws/authorize?key={self.consumer_key}&token={requests.utils.quote(request_token, safe='')}"

    def get_access_token(self, request_token: str, request_token_secret: str, verifier: str) -> OAuthTokens:
        # Need a temporary session bound to the request token.
        tmp = OAuth1Session(
            client_key=self.consumer_key,
            client_secret=self.consumer_secret,
            resource_owner_key=request_token,
            resource_owner_secret=request_token_secret,
            verifier=verifier,
            signature_method="HMAC-SHA1",
        )
        url = f"{self.base}/oauth/access_token"
        tokens = tmp.fetch_access_token(url)
        return OAuthTokens(tokens["oauth_token"], tokens["oauth_token_secret"])  # type: ignore[index]

    # -------------------------
    # Accounts / portfolio
    # -------------------------
    def list_accounts(self) -> Dict[str, Any]:
        url = f"{self.base}/v1/accounts/list.json"
        r = self._session.get(url, timeout=20)
        r.raise_for_status()
        return r.json()

    def get_portfolio(self, account_id_key: str) -> Dict[str, Any]:
        url = f"{self.base}/v1/accounts/{account_id_key}/portfolio.json"
        r = self._session.get(url, timeout=20)
        r.raise_for_status()
        return r.json()

    # -------------------------
    # Orders
    # -------------------------
    def list_orders(self, account_id_key: str, status: str = "OPEN", count: int = 50) -> Dict[str, Any]:
        url = f"{self.base}/v1/accounts/{account_id_key}/orders.json"
        r = self._session.get(url, params={"status": status, "count": count}, timeout=20)
        r.raise_for_status()
        return r.json()

    def cancel_order(self, account_id_key: str, order_id: int) -> Dict[str, Any]:
        url = f"{self.base}/v1/accounts/{account_id_key}/orders/cancel.json"
        # Per E*TRADE Order API docs, cancel is a PUT with CancelOrderRequest.
        # Keep payload strictly JSON-serializable.
        payload = {"CancelOrderRequest": {"orderId": int(order_id)}}
        r = self._session.put(url, json=payload, timeout=20)
        r.raise_for_status()
        return r.json()

    def _extract_preview_id(self, resp: Dict[str, Any]) -> Optional[int]:
        """Best-effort extraction of previewId from PreviewOrder responses."""
        if not isinstance(resp, dict):
            return None

        def _walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.lower() == "previewid":
                        try:
                            return int(v)
                        except Exception:
                            pass
                    r = _walk(v)
                    if r is not None:
                        return r
            elif isinstance(obj, list):
                for it in obj:
                    r = _walk(it)
                    if r is not None:
                        return r
            return None

        return _walk(resp)

    def preview_order(self, account_id_key: str, order: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base}/v1/accounts/{account_id_key}/orders/preview.json"
        # E*TRADE docs show Order as a LIST (even for a single order).
        payload = {
            "PreviewOrderRequest": {
                "orderType": "EQ",
                "Order": [order],
            }
        }
        r = self._session.post(url, json=payload, timeout=20)
        r.raise_for_status()
        return r.json()

    def place_order(self, account_id_key: str, order: Dict[str, Any], preview_id: Optional[int] = None) -> Dict[str, Any]:
        url = f"{self.base}/v1/accounts/{account_id_key}/orders/place.json"
        # E*TRADE docs show Order as a LIST and expect PreviewIds when placing.
        por: Dict[str, Any] = {
            "orderType": "EQ",
            "Order": [order],
        }
        if preview_id is not None:
            por["PreviewIds"] = [{"previewId": int(preview_id)}]
        payload = {"PlaceOrderRequest": por}
        r = self._session.post(url, json=payload, timeout=20)
        r.raise_for_status()
        return r.json()

    # -------------------------
    # Order builders (EQ only)
    # -------------------------
    @staticmethod
    def build_equity_order(
        symbol: str,
        action: str,
        quantity: int,
        price_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        market_session: str = "REGULAR",
        order_term: str = "GOOD_FOR_DAY",
    ) -> Dict[str, Any]:
        """Build an EQ order dict compatible with preview/place endpoints.

        Notes on time-in-force ("good for day" vs IOC/FOK):
        - For your auto-exec strategy we *explicitly* default to GOOD_FOR_DAY.
        - We do not use IOC/FOK special instructions because your bot manages
          freshness via its own timeout + EOD cleanup.
        """
        # Normalize / guard order term to avoid ambiguous broker defaults.
        # E*TRADE uses orderTerm values like GOOD_FOR_DAY / GOOD_TILL_CANCEL.
        term = str(order_term or "GOOD_FOR_DAY").upper().strip()
        if term in {"DAY", "GFD"}:
            term = "GOOD_FOR_DAY"
        elif term in {"GTC"}:
            term = "GOOD_TILL_CANCEL"
        if term not in {"GOOD_FOR_DAY", "GOOD_TILL_CANCEL"}:
            raise ValueError(f"Unsupported order_term: {order_term}")
        if price_type not in {"MARKET", "LIMIT", "STOP"}:
            raise ValueError(f"Unsupported price_type: {price_type}")
        if price_type == "LIMIT" and limit_price is None:
            raise ValueError("limit_price required for LIMIT orders")
        if price_type == "STOP" and stop_price is None:
            raise ValueError("stop_price required for STOP orders")

        order: Dict[str, Any] = {
            "orderTerm": term,
            "marketSession": market_session,
            "priceType": price_type,
            "stopPrice": float(stop_price or 0),
            "limitPrice": float(limit_price or 0),
            "allOrNone": False,
            "Instrument": [
                {
                    "Product": {"securityType": "EQ", "symbol": symbol},
                    "orderAction": action,
                    "quantityType": "QUANTITY",
                    "quantity": int(quantity),
                }
            ],
        }
        return order
    # -------------------------
    # Convenience wrappers used by auto_exec.py
    # -------------------------
    def _extract_order_id(self, resp: Dict[str, Any]) -> int:
        """Best-effort extraction of orderId from E*TRADE responses."""
        if not isinstance(resp, dict):
            raise RuntimeError(f"Unexpected response type: {type(resp)}")
        # Common keys
        for key in ("orderId", "OrderId", "orderID", "OrderID"):
            if key in resp and str(resp[key]).isdigit():
                return int(resp[key])
        # PlaceOrderResponse.OrderIds.OrderId[0].orderId
        def _walk(obj):
            if isinstance(obj, dict):
                for k,v in obj.items():
                    if k.lower() == "orderid" and isinstance(v, (int,str)) and str(v).isdigit():
                        return int(v)
                    r=_walk(v)
                    if r is not None:
                        return r
            elif isinstance(obj, list):
                for it in obj:
                    r=_walk(it)
                    if r is not None:
                        return r
            return None
        oid=_walk(resp)
        if oid is None:
            raise RuntimeError(f"Could not extract orderId from response keys={list(resp.keys())[:10]}")
        return int(oid)

    def place_equity_limit_order(
        self,
        account_id_key: str,
        symbol: str,
        qty: int,
        limit_price: float,
        action: str,
        market_session: str = "REGULAR",
    ) -> int:
        order = self.build_equity_order(
            symbol=symbol,
            action=action,
            quantity=int(qty),
            price_type="LIMIT",
            limit_price=float(limit_price),
            market_session=market_session,
            order_term="GOOD_FOR_DAY",
        )
        # Preview then place (E*TRADE requires preview before place)
        prev = self.preview_order(account_id_key, order)
        pid = self._extract_preview_id(prev)
        resp = self.place_order(account_id_key, order, preview_id=pid)
        return self._extract_order_id(resp)

    def place_equity_stop_order(
        self,
        account_id_key: str,
        symbol: str,
        qty: int,
        stop_price: float,
        action: str,
        market_session: str = "REGULAR",
    ) -> int:
        order = self.build_equity_order(
            symbol=symbol,
            action=action,
            quantity=int(qty),
            price_type="STOP",
            stop_price=float(stop_price),
            market_session=market_session,
            order_term="GOOD_FOR_DAY",
        )
        prev = self.preview_order(account_id_key, order)
        pid = self._extract_preview_id(prev)
        resp = self.place_order(account_id_key, order, preview_id=pid)
        return self._extract_order_id(resp)

    def place_equity_market_order(
        self,
        account_id_key: str,
        symbol: str,
        qty: int,
        action: str,
        market_session: str = "REGULAR",
    ) -> int:
        order = self.build_equity_order(
            symbol=symbol,
            action=action,
            quantity=int(qty),
            price_type="MARKET",
            market_session=market_session,
            order_term="GOOD_FOR_DAY",
        )
        prev = self.preview_order(account_id_key, order)
        pid = self._extract_preview_id(prev)
        resp = self.place_order(account_id_key, order, preview_id=pid)
        return self._extract_order_id(resp)

    def get_order_status_and_filled_qty(self, account_id_key: str, order_id: int) -> tuple[str, int]:
        """Return a simplified status + filled quantity for an orderId.

        Implementation uses List Orders and searches across likely statuses.
        """
        statuses = ["OPEN", "EXECUTED", "INDIVIDUAL_FILLS", "CANCELLED", "CANCEL_REQUESTED", "EXPIRED", "REJECTED"]
        found = None
        for st in statuses:
            try:
                data = self.list_orders(account_id_key, status=st, count=100)
            except Exception:
                continue
            # OrdersResponse.Orders.Order (list or dict)
            orders = (
                data.get("OrdersResponse", {})
                    .get("Orders", {})
                    .get("Order", [])
                if isinstance(data, dict) else []
            )
            if isinstance(orders, dict):
                orders = [orders]
            for o in orders:
                try:
                    oid = int(o.get("orderId", o.get("orderID", 0)))
                except Exception:
                    oid = 0
                if oid == int(order_id):
                    found = o
                    break
            if found:
                break

        if not found:
            return ("UNKNOWN", 0)

        # Normalize status
        status = str(found.get("orderStatus", found.get("status", "UNKNOWN"))).upper()
        # Best-effort filled qty extraction
        filled = 0
        # Sometimes present as filledQuantity on OrderDetail or Instrument
        def _sum_filled(obj):
            nonlocal filled
            if isinstance(obj, dict):
                for k,v in obj.items():
                    if k.lower() in {"filledquantity", "filledqty"}:
                        try:
                            filled += int(float(v))
                        except Exception:
                            pass
                    else:
                        _sum_filled(v)
            elif isinstance(obj, list):
                for it in obj:
                    _sum_filled(it)
        _sum_filled(found)
        # Fallback to executed quantity if nothing found
        if filled == 0:
            for k in ("quantityExecuted", "executedQuantity", "filled"):
                if k in found:
                    try:
                        filled = int(float(found[k]))
                        break
                    except Exception:
                        pass
        return (status, int(filled))

    def get_positions_map(self, account_id_key: str, market_session: str = "REGULAR") -> Dict[str, int]:
        """Return a simple {symbol: quantity} map for current positions."""
        data = self.get_portfolio(account_id_key)  # already .json
        pos = (
            data.get("PortfolioResponse", {})
                .get("AccountPortfolio", [])
            if isinstance(data, dict) else []
        )
        if isinstance(pos, dict):
            pos = [pos]
        out: Dict[str, int] = {}
        for ap in pos:
            positions = ap.get("Position", [])
            if isinstance(positions, dict):
                positions = [positions]
            for p in positions:
                sym = str(p.get("symbol", p.get("displaySymbol", p.get("Product", {}).get("symbol", "")))).upper()
                if not sym:
                    prod = p.get("Product", {}) if isinstance(p.get("Product", {}), dict) else {}
                    sym = str(prod.get("symbol", "")).upper()
                qty = p.get("quantity", p.get("qty", p.get("positionQuantity", 0)))
                try:
                    out[sym] = int(float(qty))
                except Exception:
                    continue
        return out

