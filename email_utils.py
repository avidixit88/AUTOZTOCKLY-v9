import smtplib
from email.message import EmailMessage
from typing import Dict, Any, List

def send_email_alert(
    smtp_server: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_emails: List[str],
    subject: str,
    body: str,
) -> None:
    """Send plaintext emails via SMTP.

    IMPORTANT: Sends *individually* (one email per recipient) so recipients do
    not see each other.
    """
    # Defensive normalization (Secrets may contain stray whitespace)
    recipients = [str(e).strip() for e in (to_emails or []) if str(e).strip()]
    if not recipients:
        return

    with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)

        for r in recipients:
            msg = EmailMessage()
            msg["From"] = smtp_user
            msg["To"] = r
            msg["Subject"] = subject
            msg.set_content(body)
            # Force single envelope recipient for privacy.
            server.send_message(msg, to_addrs=[r])


def format_alert_email(payload: Dict[str, Any]) -> str:
    """Create a human-readable email body from an alert payload dict."""
    # The app's internal row/payload objects sometimes use TitleCase keys
    # (e.g., "Symbol", "Bias", "Score", "AsOf") while other paths use
    # lowercase keys ("symbol", "bias", ...). Normalize reads so emails
    # never come through as "None".
    def g(*keys, default=None):
        for k in keys:
            if k in payload and payload.get(k) is not None:
                return payload.get(k)
        return default

    lines = []
    lines.append(f"Time: {g('time','Time','as_of','as_of','asof','AsOf')}")
    lines.append(f"Symbol: {g('symbol','Symbol')}")
    lines.append(
        f"Bias: {g('bias','Bias')}   Tier: {g('tier','Tier','stage','Stage')}   Score: {g('score','Score')}   Session: {g('session','Session')}"
    )
    lines.append("")
    lines.append(f"Last: {g('last','Last')}")
    lines.append(f"Entry (limit): {g('entry_limit','Entry','entry')}")
    lines.append(f"Chase line: {g('entry_chase_line','Chase','chase')}")

    # RIDE / continuation fields (if present)
    br = g('break_trigger','BreakTrigger','breakTrigger')
    pb = g('pullback_entry','PullbackEntry','pullback_entry')
    extras = g('extras','Extras', default={}) or {}
    family = (g('signal_family','SignalFamily','family','Family') or extras.get('family') or '').upper()
    # SWING-specific diagnostics (kept isolated so other families' emails remain identical)
    if family == 'SWING':
        swing_stage = extras.get('swing_stage') or g('swing_stage','SwingStage')
        if swing_stage:
            lines.append(f"Swing stage: {swing_stage}")
        tls = extras.get('trend_lock_score')
        if tls is not None:
            lines.append(f"Trend lock: {tls}/5")
        retr = extras.get('retrace_pct')
        if retr is not None:
            try:
                lines.append(f"Retrace: {float(retr):.1f}%")
            except Exception:
                lines.append(f"Retrace: {retr}")
        im_start = extras.get('impulse_start')
        im_end = extras.get('impulse_end')
        rmode = extras.get('retrace_mode')
        if im_start is not None and im_end is not None:
            try:
                lines.append(f"Impulse leg: {float(im_start):.4f} → {float(im_end):.4f}" + (f" ({rmode})" if rmode else ""))
            except Exception:
                lines.append(f"Impulse leg: {im_start} → {im_end}" + (f" ({rmode})" if rmode else ""))
        pbq = extras.get('pullback_quality')
        pbq_r = extras.get('pullback_quality_reasons')
        if pbq is not None:
            lines.append(f"Pullback quality: {pbq}/6" + (f" ({pbq_r})" if pbq_r else ""))
        conf_n = extras.get('confluence_count')
        conf = extras.get('confluences')
        if conf_n is not None:
            lines.append(f"Confluence: {conf_n}" + (f" ({conf})" if conf else ""))
        ez = extras.get('entry_zone')
        if ez:
            lines.append(f"Entry zone: {ez}")
        etr = extras.get('entry_trigger_reason')
        if etr:
            lines.append(f"Entry trigger: {etr}")
        # Allow SWING pullback band to display using extras['pullback_band'] if pb1/pb2 are absent
        if g('pb1','PB1','pb1') is None and g('pb2','PB2','pb2') is None:
            pb_band = extras.get('pullback_band')
            if isinstance(pb_band, (tuple, list)) and len(pb_band) == 2:
                try:
                    pb1 = float(min(pb_band))
                    pb2 = float(max(pb_band))
                    extras['pb1'] = pb1
                    extras['pb2'] = pb2
                except Exception:
                    pass

    # HEAVENLY-specific diagnostics (isolated so other families remain unchanged)
    if family == 'HEAVENLY':
        stage = g('stage','Stage','tier','Tier') or extras.get('stage')
        if stage:
            lines.append(f"Heavenly stage: {stage}")
        tsz = extras.get('tsz')
        if tsz:
            lines.append(f"TSZ: {tsz}")
        w_atr = extras.get('tsz_width_atr')
        if w_atr is not None:
            try:
                lines.append(f"TSZ width: {float(w_atr):.2f} ATR")
            except Exception:
                lines.append(f"TSZ width: {w_atr}")
        con = extras.get('tsz_constraints')
        if con:
            lines.append(f"Constraints: {con}")
        evs = extras.get('evs')
        if evs is not None:
            try:
                lines.append(f"EVS: {float(evs):.2f} ATR")
            except Exception:
                lines.append(f"EVS: {evs}")
        ob_t = extras.get('evs_obstacle')
        ob_p = extras.get('evs_obstacle_price')
        if ob_t and ob_p is not None:
            try:
                lines.append(f"Next obstacle: {ob_t} @ {float(ob_p):.4f}")
            except Exception:
                lines.append(f"Next obstacle: {ob_t} @ {ob_p}")
        trig = extras.get('trigger_type')
        if trig:
            lines.append(f"Trigger: {trig}")
        il = extras.get('intent_label')
        iscore = extras.get('intent_score')
        if il:
            if iscore is not None:
                lines.append(f"1m intent: {il} ({iscore})")
            else:
                lines.append(f"1m intent: {il}")
    pb1 = g('pb1','PB1','pb1')
    pb2 = g('pb2','PB2','pb2')
    if pb1 is None:
        pb1 = extras.get('pb1')
    if pb2 is None:
        pb2 = extras.get('pb2')
    if pb is not None:
        lines.append(f"Pullback entry: {pb}")
    if pb1 is not None and pb2 is not None:
        lines.append(f"Pullback band: {pb1} – {pb2}")
    if br is not None:
        lines.append(f"Break trigger: {br}")

    lines.append(f"Stop: {g('stop','Stop')}")
    lines.append(f"TP0: {g('tp0','TP0')}")
    # Support both canonical keys (tp1/tp2) and internal short keys (t1/t2)
    lines.append(f"TP1: {g('tp1','TP1','t1','T1')}")
    lines.append(f"TP2: {g('tp2','TP2','t2','T2')}")
    lines.append(f"TP3: {g('tp3','TP3','t3','T3')}")
    lines.append(f"ETA TP0 (min): {g('eta_tp0_min','ETA TP0 (min)')}")
    lines.append("")
    why = g('why','Why', default="") or ""
    lines.append("Why:")
    lines.append(str(why))
    lines.append("")
    # re-use extras collected above
    if extras:
        lines.append("Diagnostics:")
        for k in [
            "vwap_logic",
            "session_vwap_include_premarket",
            "liquidity_phase",
            "accept_line",
            "impulse_quality",
            "disp_ratio",
            "atr_pct",
            "baseline_atr_pct",
            "atr_ref_pct",
            "atr_score_scale",
            "htf_bias",
        ]:
            if k in extras:
                lines.append(f"- {k}: {extras.get(k)}")
    return "\n".join(lines)