import re
import pandas as pd


def _extract_user_susceptibility(description: str) -> float:
    """
    从用户描述里抽一个弱风险分：
    - 明确百分比 -> 按百分比
    - 提到容易相信可疑消息 / phishing -> 给较高分
    - 默认中性值
    """
    if not isinstance(description, str):
        return 0.35

    text = description.lower()

    m = re.search(r"(\d+)\s*%", text)
    if m:
        pct = float(m.group(1)) / 100.0
        return min(max(pct, 0.0), 1.0)

    risky_patterns = [
        "phishing",
        "courriels douteux",
        "suspicious link",
        "trust",
        "confiance",
        "online lure",
    ]
    if any(p in text for p in risky_patterns):
        return 0.65

    return 0.35


def enrich_with_user_profiles(tx: pd.DataFrame, users: pd.DataFrame | None) -> pd.DataFrame:
    out = tx.copy()

    if users is None or users.empty:
        # 填默认列，避免后面报错
        for col in [
            "_sender_salary",
            "_sender_user_risk",
            "_sender_home_city",
            "_recipient_salary",
            "_recipient_user_risk",
            "_recipient_home_city",
        ]:
            out[col] = 0
        return out

    u = users.copy()

    # 统一字段
    if "description" not in u.columns:
        u["description"] = ""

    u["_user_risk"] = u["description"].apply(_extract_user_susceptibility)

    sender_user = u.rename(columns={
        "iban": "sender_iban",
        "salary": "_sender_salary",
        "job": "_sender_job",
        "residence_city": "_sender_home_city",
        "residence_lat": "_sender_home_lat",
        "residence_lng": "_sender_home_lng",
        "_user_risk": "_sender_user_risk",
        "first_name": "_sender_first_name",
        "last_name": "_sender_last_name",
    })

    recipient_user = u.rename(columns={
        "iban": "recipient_iban",
        "salary": "_recipient_salary",
        "job": "_recipient_job",
        "residence_city": "_recipient_home_city",
        "residence_lat": "_recipient_home_lat",
        "residence_lng": "_recipient_home_lng",
        "_user_risk": "_recipient_user_risk",
        "first_name": "_recipient_first_name",
        "last_name": "_recipient_last_name",
    })

    sender_keep = [
        "sender_iban",
        "_sender_salary",
        "_sender_job",
        "_sender_home_city",
        "_sender_home_lat",
        "_sender_home_lng",
        "_sender_user_risk",
        "_sender_first_name",
        "_sender_last_name",
    ]
    recipient_keep = [
        "recipient_iban",
        "_recipient_salary",
        "_recipient_job",
        "_recipient_home_city",
        "_recipient_home_lat",
        "_recipient_home_lng",
        "_recipient_user_risk",
        "_recipient_first_name",
        "_recipient_last_name",
    ]

    out = out.merge(sender_user[sender_keep], how="left", on="sender_iban")
    out = out.merge(recipient_user[recipient_keep], how="left", on="recipient_iban")

    for col in [
        "_sender_salary",
        "_sender_user_risk",
        "_recipient_salary",
        "_recipient_user_risk",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    return out