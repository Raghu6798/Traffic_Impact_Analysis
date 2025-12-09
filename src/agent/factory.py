from langchain.agents import create_agent
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import (
    ToolMessage,
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_google_genai._function_utils import convert_to_genai_function_declarations
from google.ai.generativelanguage_v1beta.types import Tool as GoogleTool
from typing import Any, Dict, List, Optional, Sequence, Union, Callable, Literal
from pydantic import SecretStr
import json
import requests
import httpx
from langchain_google_genai._function_utils import tool_to_dict

class ChatOpenAI(BaseChatModel):
    """
    A custom LangChain chat model that can connect to any OpenAI-compatible API endpoint.
    Includes provider-specific logic for handling tool binding and payload creation.
    """
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: SecretStr
    base_url: str

    @property
    def _llm_type(self) -> str:
        return "openai_compatible_chat"

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, "BaseTool"]],
        *,
        tool_choice: Optional[Union[str, bool, Literal["auto", "any"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """
        Bind tools to the model, applying provider-specific logic.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        if self.provider_id == "google":
            try:
                genai_tools = tool_to_dict(convert_to_genai_function_declarations(tools))
                kwargs["tools"] = genai_tools["function_declarations"]
            except Exception:
                kwargs["tools"] = formatted_tools
        else:
            kwargs["tools"] = formatted_tools

        if tool_choice is not None:
            if self.provider_id == "groq" and tool_choice == "any":
                kwargs["tool_choice"] = "required"
            elif isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
                kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
            else:
                kwargs["tool_choice"] = tool_choice
        
        return self.bind(**kwargs)

     def _create_payload(self, messages: List[BaseMessage], stop: Optional[List[str]], **kwargs: Any) -> Dict[str, Any]:
        """Creates the JSON payload for the API request, with provider-specific adjustments."""
        tool_call_map = {tc["id"]: tc["name"] for m in messages if isinstance(m, AIMessage) and m.tool_calls for tc in m.tool_calls}

        message_dicts = []
        for m in messages:
            if isinstance(m, HumanMessage):
                message_dicts.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                msg = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    msg["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}} for tc in m.tool_calls]
                message_dicts.append(msg)
            elif isinstance(m, SystemMessage):
                message_dicts.append({"role": "system", "content": m.content})
            elif isinstance(m, ToolMessage):
                tool_msg = {"role": "tool", "content": m.content, "tool_call_id": m.tool_call_id}
                if self.provider_id == "cerebras":
                    tool_name = tool_call_map.get(m.tool_call_id)
                    if tool_name: tool_msg["name"] = tool_name
                    else: logger.warning(f"Could not find name for tool_call_id: {m.tool_call_id}")
                message_dicts.append(tool_msg)
        
        payload = {"model": self.model, "messages": message_dicts, "temperature": self.temperature, **kwargs}
        if self.max_tokens: payload["max_tokens"] = self.max_tokens
        if stop: payload["stop"] = stop

        if self.provider_id == "cerebras" and "tools" in payload:
            payload["parallel_tool_calls"] = False
            logger.debug("Set parallel_tool_calls=False for Cerebras provider.")
            
        return payload

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous call to an OpenAI-compatible API."""
        payload = self._create_payload(messages, stop=stop, **kwargs)
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }


        try:
            logger.info(f"The base_url being hit is {self.base_url} , payload : {payload} and {headers}")
            response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

        # 3. Parse the response into a LangChain ChatResult
        return self._parse_response(data)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous call to an OpenAI-compatible API."""
        # 1. Prepare messages and payload
        payload = self._create_payload(messages, stop=stop, **kwargs)
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        # 2. Make the async HTTP request
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0, # Add a reasonable timeout
                )
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            logger.error(f"Async API call failed: {e}")
            raise

        # 3. Parse the response into a LangChain ChatResult
        return self._parse_response(data)
    
    def _create_payload(self, messages: List[BaseMessage], stop: Optional[List[str]], **kwargs: Any) -> Dict[str, Any]:
        """Creates the JSON payload for the API request."""
        message_dicts = []
        for m in messages:
            if isinstance(m, HumanMessage):
                message_dicts.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                msg = {"role": "assistant", "content": m.content or ""}

                if m.tool_calls:
                
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function", 
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                        }
                        for tc in m.tool_calls
                    ]
                message_dicts.append(msg)
            elif isinstance(m, SystemMessage):
                message_dicts.append({"role": "system", "content": m.content})
            elif isinstance(m, ToolMessage):
                message_dicts.append({
                    "role": "tool",
                    "content": m.content,
                    "tool_call_id": m.tool_call_id,
                })
        
        # Build final payload
        payload = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
            **kwargs,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if stop:
            payload["stop"] = stop
            
        return payload

    def _parse_response(self, data: Dict[str, Any]) -> ChatResult:
        """Parses the API JSON response into a ChatResult."""
        choice = data["choices"][0]
        message_data = choice["message"]
        
        content = message_data.get("content", "")
        
        tool_calls = []
        if message_data.get("tool_calls"):
            for tc in message_data["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {} 
                
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "args": args,
                })

        ai_message = AIMessage(
            content=content or "",
            tool_calls=tool_calls if tool_calls else []
        )
        
        return ChatResult(generations=[ChatGeneration(message=ai_message)])
