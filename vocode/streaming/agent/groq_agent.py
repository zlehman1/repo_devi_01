import os
from groq import Groq
from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.models.agent import GroqAgentConfigType, GeneratedResponse, BaseMessage, EndOfTurn
from vocode.streaming.utils import logger
from vocode.streaming.action.default_factory import DefaultActionFactory
from vocode.streaming.vector_db.factory import VectorDBFactory
from vocode.streaming.action.abstract_factory import AbstractActionFactory
from typing import AsyncGenerator

class GroqAgent(RespondAgent[GroqAgentConfigType]):
    groq_client: Groq

    def __init__(
        self,
        agent_config: GroqAgentConfigType,
        action_factory: AbstractActionFactory = DefaultActionFactory(),
        vector_db_factory=VectorDBFactory(),
        **kwargs,
    ):
        super().__init__(
            agent_config=agent_config,
            action_factory=action_factory,
            **kwargs,
        )
        self.groq_client = Groq(api_key=agent_config.groq_api_key or os.environ["GROQ_API_KEY"])

        if not self.groq_client.api_key:
            raise ValueError("GROQ_API_KEY must be set in environment or passed in")

        if self.agent_config.vector_db_config:
            self.vector_db = vector_db_factory.create_vector_db(self.agent_config.vector_db_config)

    async def generate_response(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
    ) -> AsyncGenerator[GeneratedResponse, None]:
        chat_parameters = {
            "messages": [
                {"role": "system", "content": "you are a helpful assistant."},
                {"role": "user", "content": human_input}
            ],
            "model": self.agent_config.model_name,
            "stream": True
        }

        if self.agent_config.vector_db_config:
            try:
                docs_with_scores = await self.vector_db.similarity_search_with_score(
                    self.transcript.get_last_user_message()[1]
                )
                docs_with_scores_str = "\n\n".join(
                    [
                        "Document: "
                        + doc[0].metadata["source"]
                        + f" (Confidence: {doc[1]})\n"
                        + doc[0].lc_kwargs["page_content"].replace(r"\n", "\n")
                        for doc in docs_with_scores
                    ]
                )
                vector_db_result = (
                    f"Found {len(docs_with_scores)} similar documents:\n{docs_with_scores_str}"
                )
                chat_parameters["messages"].insert(-1, {"role": "system", "content": vector_db_result})
            except Exception as e:
                logger.error(f"Error while hitting vector db: {e}", exc_info=True)

        try:
            stream = await self.groq_client.chat.completions.create(**chat_parameters)
            async for message in stream:
                yield GeneratedResponse(
                    response=message["content"],
                    confidence=None,
                )
        except Exception as e:
            logger.error(f"Error while generating response: {e}", exc_info=True)
            yield GeneratedResponse(
                response="I'm sorry, I encountered an error.",
                confidence=None,
            )
