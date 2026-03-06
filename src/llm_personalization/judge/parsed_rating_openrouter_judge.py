import asyncio
import os
import re
from openai import AsyncOpenAI, APIStatusError
from tqdm.asyncio import tqdm

from .judge import PrincipleJudge
from .parsed_rating_judge import JUDGE_SYSTEM_PROMPT_TEMPLATE, JUDGE_USER_TEMPLATE


class ParsedRatingOpenRouterJudge(PrincipleJudge):
    def __init__(self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        judge_system_prompt_template: str = JUDGE_SYSTEM_PROMPT_TEMPLATE,
        judge_user_template: str = JUDGE_USER_TEMPLATE,
        enable_thinking: bool = False,
        reasoning_max_tokens: int | None = None,
        max_tokens: int = 4,
        temperature: float = 0.0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        top_k: int | None = None,
        min_p: float | None = None,
        max_concurrent_requests: int = 50,
        request_delay: float = 0.0,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.judge_system_prompt_template = judge_system_prompt_template
        self.judge_user_template = judge_user_template
        self.enable_thinking = enable_thinking
        self.reasoning_max_tokens = reasoning_max_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.max_concurrent_requests = max_concurrent_requests
        self.request_delay = request_delay

    def load(self) -> None:
        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No API key provided and OPENROUTER_API_KEY env var is not set.")
        self.client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    def unload(self) -> None:
        self.client = None

    def _parse_score(self, text: str) -> int | None:
        text = text.strip()
        if self.enable_thinking:
            matches = re.findall(r'\b(10|[1-9])\b', text)
            if matches:
                return int(matches[-1])
            return None
        if not text.isdigit():
            return None
        score = int(text)
        if not (1 <= score <= 10):
            return None
        return score

    @staticmethod
    def _extract_text_from_content(content) -> str:
        """Handle both plain string and Claude's block-list content format."""
        if isinstance(content, list):
            return next((b["text"] for b in content if b.get("type") == "text"), "")
        return content or ""

    async def _call_one(self, semaphore: asyncio.Semaphore, messages: list[dict], max_api_retries: int = 6) -> str:
        async with semaphore:
            if self.request_delay > 0:
                await asyncio.sleep(self.request_delay)
            kwargs = dict(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
            )
            extra_body = {}
            reasoning_body = {"enabled": self.enable_thinking}
            if self.top_k is not None:
                extra_body["top_k"] = self.top_k
            if self.min_p is not None:
                extra_body["min_p"] = self.min_p
            if self.enable_thinking and self.reasoning_max_tokens is not None:
                reasoning_body["max_tokens"] = self.reasoning_max_tokens
            extra_body["reasoning"] = reasoning_body
            kwargs["extra_body"] = extra_body

            for attempt in range(max_api_retries + 1):
                try:
                    response = await self.client.chat.completions.create(**kwargs)
                    return self._extract_text_from_content(response.choices[0].message.content)
                except APIStatusError as e:
                    if attempt == max_api_retries:
                        raise
                    wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80, 160s
                    print(f"\n  HTTP {e.status_code} from provider, retrying in {wait}s (attempt {attempt + 1}/{max_api_retries})...")
                    await asyncio.sleep(wait)

    async def _call_one_with_progress(self, semaphore: asyncio.Semaphore, messages: list[dict], pbar: tqdm) -> str:
        result = await self._call_one(semaphore, messages)
        pbar.update(1)
        return result

    async def _judge_manual_async(self, all_messages: list[list[dict]], max_retries: int = 4) -> list[int]:
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        with tqdm(total=len(all_messages), desc=f"Judging ({self.model})") as pbar:
            tasks = [self._call_one_with_progress(semaphore, msgs, pbar) for msgs in all_messages]
            texts = list(await asyncio.gather(*tasks))
        scores: list[int | None] = [self._parse_score(t) for t in texts]

        for attempt in range(max_retries):
            failed_indices = [i for i, s in enumerate(scores) if s is None]
            if not failed_indices:
                break
            print(f"Retrying {len(failed_indices)} failed request(s) (attempt {attempt + 1}/{max_retries})...")
            for i in failed_indices:
                print(f"  [idx {i}] bad output: {texts[i]!r}")
            with tqdm(total=len(failed_indices), desc=f"Retry {attempt + 1}/{max_retries}") as pbar:
                retry_tasks = [self._call_one_with_progress(semaphore, all_messages[i], pbar) for i in failed_indices]
                retry_texts = await asyncio.gather(*retry_tasks)
            for i, text in zip(failed_indices, retry_texts):
                texts[i] = text
                scores[i] = self._parse_score(text)

        for i, s in enumerate(scores):
            if s is None:
                print(f"Warning: giving up on idx {i} after {max_retries} retries, output: {texts[i]!r}, returning 5")
                scores[i] = 5

        return scores

    def judge_manual(self, all_messages: list[list[dict]]) -> list[int]:
        return asyncio.run(self._judge_manual_async(all_messages))

    def judge_principle(self, conversations: list[list[dict[str, str]]], principles: list[str]) -> list[int]:
        all_messages = []

        for messages, principle in zip(conversations, principles):
            if len(messages) < 2:
                raise ValueError(f"Conversation has less than 2 messages: {messages}")
            if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
                raise ValueError("Last message must be an assistant response and the second to last must be a user prompt.")
            message_string = ""
            for message in messages[:-1]:
                message_string += f"<message role='{message['role']}'>{message['content']}</message>\n"
            system_prompt = self.judge_system_prompt_template
            user_prompt = self.judge_user_template.format(
                conversation=message_string,
                response=messages[-1]["content"],
                principle=principle,
            )
            all_messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

        return self.judge_manual(all_messages)
