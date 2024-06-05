from typing import TYPE_CHECKING

from vocode.streaming.agent.abstract_factory import AbstractAgentFactory
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.agent.echo_agent import EchoAgent
from vocode.streaming.agent.restful_user_implemented_agent import RESTfulUserImplementedAgent
from vocode.streaming.models.agent import (
    AgentConfig,
    AnthropicAgentConfig,
    ChatGPTAgentConfig,
    EchoAgentConfig,
    RESTfulUserImplementedAgentConfig,
)

if TYPE_CHECKING:
    from vocode.streaming.agent.anthropic_agent import AnthropicAgent
    from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent


class DefaultAgentFactory(AbstractAgentFactory):
    def create_agent(self, agent_config: AgentConfig) -> BaseAgent:
        if isinstance(agent_config, ChatGPTAgentConfig):
            return self._get_chat_gpt_agent(agent_config)
        elif isinstance(agent_config, EchoAgentConfig):
            return EchoAgent(agent_config=agent_config)
        elif isinstance(agent_config, RESTfulUserImplementedAgentConfig):
            return RESTfulUserImplementedAgent(agent_config=agent_config)
        elif isinstance(agent_config, AnthropicAgentConfig):
            return self._get_anthropic_agent(agent_config)
        raise Exception("Invalid agent config", agent_config.type)

    def _get_chat_gpt_agent(self, agent_config: ChatGPTAgentConfig) -> "ChatGPTAgent":
        try:
            from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent

            return ChatGPTAgent(agent_config=agent_config)
        except ImportError as e:
            raise ImportError(f"Missing required dependancies for Agent {agent_config.type}") from e

    def _get_anthropic_agent(self, agent_config: AnthropicAgentConfig) -> "AnthropicAgent":
        try:
            from vocode.streaming.agent.anthropic_agent import AnthropicAgent

            return AnthropicAgent(agent_config=agent_config)
        except ImportError as e:
            raise ImportError(f"Missing required dependancies for Agent {agent_config.type}") from e
