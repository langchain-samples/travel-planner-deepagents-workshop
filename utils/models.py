"""Centralized chat model factory for the workshop.

Auto-detects Azure OpenAI when AZURE_OPENAI_ENDPOINT is set in the environment;
otherwise falls back to standard OpenAI via init_chat_model. Switch providers
by editing .env, not by editing this file.

Required env vars per mode:
    Azure (when AZURE_OPENAI_ENDPOINT is set):
        AZURE_OPENAI_API_KEY
        AZURE_OPENAI_ENDPOINT
        AZURE_OPENAI_DEPLOYMENT
        AZURE_OPENAI_API_VERSION    (optional, defaults to 2025-03-01-preview)
    OpenAI (default):
        OPENAI_API_KEY

Bedrock / Vertex examples are kept commented below for reference.
"""

import os

from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env", override=True)

from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI

DEFAULT_OPENAI_MODEL = "openai:gpt-5.4"


def _azure_mode() -> bool:
    """True if Azure OpenAI env vars indicate we should use AzureChatOpenAI."""
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT"))


def get_model(name: str | None = None):
    """Return a chat model based on the active provider mode.

    In Azure mode the `name` argument is ignored because Azure routes by
    deployment, not by model name. Set AZURE_OPENAI_DEPLOYMENT to target a
    different deployment.

    In OpenAI mode, `name` defaults to DEFAULT_OPENAI_MODEL.
    """
    if _azure_mode():
        required = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT")
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            raise RuntimeError(
                f"Azure mode detected (AZURE_OPENAI_ENDPOINT is set) but missing required env vars: {missing}. "
                "Either set them in .env, or unset AZURE_OPENAI_ENDPOINT to fall back to standard OpenAI."
            )
        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2025-03-01-preview"
            ),
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
        )
    return init_chat_model(name or DEFAULT_OPENAI_MODEL)


# Main agent model. Imported throughout the workshop as `from utils.models import model`.
model = get_model()


"""Bedrock Version"""
# from dotenv import load_dotenv
# from langchain_aws import ChatBedrockConverse
# import os

# load_dotenv(dotenv_path="../.env", override=True)

# AWS_ACCESS_KEY_ID=os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY=os.getenv("AWS_SECRET_ACCESS_KEY")
# AWS_REGION_NAME=os.getenv("AWS_REGION_NAME")
# AWS_MODEL_ARN=os.getenv("AWS_MODEL_ARN")

# model = ChatBedrockConverse(
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#     region_name=AWS_REGION_NAME,
#     provider="anthropic",
#     model_id=AWS_MODEL_ARN
# )


"""Google Vertex AI version"""
# Make sure you have your vertex ai credentials setup and your GOOGLE_APPLICATION_CREDENTIALS are pointing to the JSON file.

# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model

# # Find project root and load .env
# project_root = Path(__file__).resolve().parent.parent
# load_dotenv(dotenv_path=project_root / ".env", override=True)

# # Fix credentials path to absolute
# if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
#     cred_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
#     if not os.path.isabs(cred_path):
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_root / cred_path.lstrip("./"))

# # Create model
# model = init_chat_model("google_vertexai:gemini-2.5-flash")
