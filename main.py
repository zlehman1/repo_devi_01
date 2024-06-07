import os
import sys
import nltk
import datetime
import pytz
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from loguru import logger
from pyngrok import ngrok
import uvicorn

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

from speller_agent import SpellerAgentFactory
from vocode.logging import configure_pretty_logging
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.config_manager.redis_config_manager import RedisConfigManager
from vocode.streaming.telephony.server.base import TelephonyServer, TwilioInboundCallConfig
from vocode.streaming.synthesizer.eleven_labs_websocket_synthesizer import ElevenLabsWSSynthesizer
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.transcriber import TranscriberConfig, DeepgramTranscriberConfig
from memory_config import config_manager

nltk.download('punkt')

load_dotenv()
configure_pretty_logging()

app = FastAPI(docs_url=None)
config_manager = RedisConfigManager()

BASE_URL = os.getenv("BASE_URL")

SYNTH_CONFIG = ElevenLabsSynthesizerConfig.from_telephone_output_device(api_key=os.getenv("ELEVEN_LABS_API_KEY"), voice_id="HPVJxohvv28Bwatdj5rG")


def get_greeting():
    pacific_tz = pytz.timezone('America/Los_Angeles')
    current_hour = datetime.datetime.now(pacific_tz).hour
    if current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"


greeting = get_greeting()

telephony_server = TelephonyServer(
    base_url=BASE_URL,
    config_manager=config_manager,
    inbound_call_configs=[
        TwilioInboundCallConfig(
            url="/inbound_call",
            agent_config=ChatGPTAgentConfig(
                initial_message=BaseMessage(text=f"{greeting}, how can I assist you today?"),
                prompt_preamble=f"{greeting}, You are a helpful Healthcare receptionist who is answering the phone. You respond with no more than 10 words. Sometimes only one or two words are needed, sometimes more. Your goal is to assist patients in navigating healthcare services and scheduling appointments. You may ask where the patient is located and if they would like to schedule an appointment at one of our healthcare facilities in Central California. We do not serve patients outside of the California Central Valley. We offer a variety of services, including primary care, specialist consultations, and lab tests. Ask the patient what time they would like to schedule an appointment. We adapt to health plan benefits and coverage. Offer information on prescription refills and adherence. Provide options for lab test scheduling and results. Provide facility contact details for inquiries.",
                generate_responses=True,
            ),
            synthesizer_config=ElevenLabsSynthesizerConfig.from_telephone_output_device(
                api_key=os.getenv("ELEVEN_LABS_API_KEY"),
                voice_id="HPVJxohvv28Bwatdj5rG",
                experimental_websocket=True
            ),
            twilio_config=TwilioConfig(
                account_sid=os.environ["TWILIO_ACCOUNT_SID"],
                auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            ),
        )
    ],
    agent_factory=SpellerAgentFactory(),
)

app.include_router(telephony_server.get_router())

@app.get("/")
async def main():
    return {"hello": "world", "base_URL": BASE_URL}


@app.get("/items/{item}")
async def subpage(item: str):
    return {"item": item}


if __name__ == "__main__":
    uvicorn.run(app, port=3000, host="0.0.0.0")
