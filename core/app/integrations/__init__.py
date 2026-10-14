# Importing these registers them into ADAPTER_REGISTRY (see base.py's register_adapter).
from app.integrations.adapters import (  # noqa: F401,E402
    custom_api, google_calendar, hubspot, internal_api, recharge, shippo, shopify, stripe, twilio, zendesk)
