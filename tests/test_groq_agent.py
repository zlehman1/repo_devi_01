import os
import pytest
from unittest.mock import AsyncMock, patch
from vocode.streaming.agent.groq_agent import GroqAgent, GroqAgentConfigType
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.agent import GeneratedResponse

@pytest.fixture
def groq_agent_config():
    return GroqAgentConfigType(
        groq_api_key="test_api_key",
        model_name="llama3-8b-8192",
        vector_db_config=None
    )

@pytest.fixture
def groq_agent(groq_agent_config):
    return GroqAgent(agent_config=groq_agent_config)

@pytest.mark.asyncio
async def test_generate_response(groq_agent):
    human_input = "Explain the importance of fast language models"
    conversation_id = "test_conversation_id"

    with patch("vocode.streaming.agent.groq_agent.Groq") as MockGroq:
        mock_groq_client = MockGroq.return_value
        mock_groq_client.chat.completions.create = AsyncMock(return_value=AsyncMock(__aiter__=lambda s: iter([{"content": "Fast language models are important because..."}])))

        responses = []
        async for response in groq_agent.generate_response(human_input, conversation_id):
            responses.append(response)

        assert len(responses) > 0
        assert isinstance(responses[0], GeneratedResponse)
        assert responses[0].response == "Fast language models are important because..."

@pytest.mark.asyncio
async def test_generate_response_error_handling(groq_agent):
    human_input = "Explain the importance of fast language models"
    conversation_id = "test_conversation_id"

    with patch("vocode.streaming.agent.groq_agent.Groq") as MockGroq:
        mock_groq_client = MockGroq.return_value
        mock_groq_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        responses = []
        async for response in groq_agent.generate_response(human_input, conversation_id):
            responses.append(response)

        assert len(responses) > 0
        assert isinstance(responses[0], GeneratedResponse)
        assert responses[0].response == "I'm sorry, I encountered an error."
