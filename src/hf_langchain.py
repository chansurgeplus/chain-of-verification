from langchain.chat_models.base import BaseChatModel
from typing import Dict, Any
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
    LLMResult,
)
from typing import Any, List, Optional, Union


DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""
DEFAULT_SYSTEM_PROMPT_RESPONSE = """Hello! how can I help you?"""

class ChatHuggingFace(BaseChatModel):
    """
    Wrapper for using Hugging Face LLM's as ChatModels.
    """

    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    system_response_ai_message: AIMessage = AIMessage(content=DEFAULT_SYSTEM_PROMPT_RESPONSE)

    tokenizer: Any = None
    hf_pipe: Any = None
    model_kwargs: Any = None

    def __init__(self, model_id: str, **kwargs: Any):
        super().__init__(**kwargs)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=kwargs["model_kwargs"]["low_cpu_mem_usage"],
            device_map=kwargs["model_kwargs"]["device_map"] or None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer, max_new_tokens=kwargs["model_kwargs"]["max_new_tokens"])
        self.hf_pipe = HuggingFacePipeline(pipeline=pipe)

        self.model_kwargs = kwargs["model_kwargs"]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = self.hf_pipe(llm_input)
        return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("at least one HumanMessage must be provided")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("last message must be a HumanMessage")

        if len(messages) > 1 and type(messages[0]) == SystemMessage:
          if type(messages[1]) == SystemMessage or type(messages[1]) == HumanMessage:
            messages.insert(1, self.system_response_ai_message)

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=self.model_kwargs["add_generation_prompt"] or False
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: str) -> ChatResult:
        chat_generations = [
            ChatGeneration(message=AIMessage(content=llm_result), generation_info=None)
        ]

        return ChatResult(
            generations=chat_generations, llm_output={
                "content": llm_result
            }
        )

    @property
    def _llm_type(self) -> str:
        return "huggingface-chat-wrapper"
