"""OpenAI Responses API 도구 루프.

스텝·시간 상한을 넘기면 그때까지의 내용으로 답한다. 여기서는 Slack도
도구 구현도 모르고, 오직 대화와 도구 호출만 다룬다.
"""
import json
import time
from typing import NamedTuple

from api.logger import logger


class AgentResult(NamedTuple):
    text: str
    steps: int
    truncated: bool
    tool_calls: list


def _as_input_item(item):
    """SDK 응답 객체를 다음 요청의 input 항목으로 되돌린다."""
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return item


def run_agent(
    *,
    client,
    model,
    system_prompt,
    user_input,
    tool_specs,
    dispatch,
    max_steps=8,
    deadline_sec=90,
    now=time.time,
    on_step=None,
) -> AgentResult:
    started = now()
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    tool_calls, steps, truncated, text = [], 0, False, ""

    while steps < max_steps:
        steps += 1
        response = client.responses.create(
            model=model, input=conversation, tools=tool_specs
        )
        calls = [i for i in response.output if getattr(i, "type", "") == "function_call"]
        if not calls:
            text = getattr(response, "output_text", "") or ""
            break

        conversation += [_as_input_item(i) for i in response.output]
        for c in calls:
            try:
                args = (
                    json.loads(c.arguments)
                    if isinstance(c.arguments, str)
                    else (c.arguments or {})
                )
            except ValueError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(c.name)
            if on_step:
                on_step(c.name)
            output = dispatch(c.name, args)
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": c.call_id,
                    "output": output,
                }
            )

        if now() - started > deadline_sec:
            truncated = True
            logger.info(f"agent loop deadline 초과: {steps} 스텝, 도구 {tool_calls}")
            break
    else:
        truncated = True
        logger.info(f"agent loop max_steps 도달: 도구 {tool_calls}")

    return AgentResult(
        text=text, steps=steps, truncated=truncated, tool_calls=tool_calls
    )
