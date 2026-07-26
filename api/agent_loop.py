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


# 서버가 응답에만 붙이는 필드. 그대로 되보내면 모델에 따라
# "Unknown parameter: input[N].status"로 400을 낸다(gpt-5.6-luna에서 확인).
_OUTPUT_ONLY_KEYS = ("status",)


def _as_input_item(item):
    """SDK 응답 객체를 다음 요청의 input 항목으로 되돌린다.

    Responses API에 previous_response_id를 넘기지 않으므로 서버는 앞 턴을
    기억하지 않는다. 모델이 만든 출력(도구 호출 포함)을 우리가 직접 다음
    input에 다시 실어야 대화가 이어진다. SDK 객체는 그대로 못 보내고
    dict으로 풀어야 한다(테스트의 가짜 객체는 dict을 그대로 준다).
    """
    data = item.model_dump() if hasattr(item, "model_dump") else item
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if k not in _OUTPUT_ONLY_KEYS}
    return data


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
        # web_search 같은 내장 도구는 서버가 알아서 실행하고 결과만 함께
        # 온다. 우리가 할 일은 없지만 무엇을 했는지는 로그에 남긴다.
        for i in response.output:
            kind = getattr(i, "type", "")
            if kind.endswith("_call") and kind != "function_call":
                tool_calls.append(kind.removesuffix("_call"))

        # 처리할 함수 호출이 없으면 그게 최종 답이다.
        calls = [i for i in response.output if getattr(i, "type", "") == "function_call"]
        if not calls:
            text = getattr(response, "output_text", "") or ""
            break

        # 도구 호출만이 아니라 출력 전체를 되싣는다. 추론 항목 등을 빼먹으면
        # 다음 요청에서 call_id 짝이 맞지 않는다.
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
            # 인자가 깨져도 여기서 죽지 않는다. 빈 dict을 넘기면 dispatch가
            # "빠진 인자" 오류를 돌려주고, 모델이 그걸 보고 다시 부른다.
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

        # 시간 확인은 도구를 다 돌린 뒤에 한다. 중간에 끊으면 이미 게시된
        # 요약과 모델이 아는 상태가 어긋난다.
        if now() - started > deadline_sec:
            truncated = True
            logger.info(f"agent loop deadline 초과: {steps} 스텝, 도구 {tool_calls}")
            break
    else:
        # while이 break 없이 끝났다 = max_steps를 다 썼는데도 도구만 부르고
        # 있었다. 이때 text는 빈 문자열이고 호출부가 폴백을 태운다.
        truncated = True
        logger.info(f"agent loop max_steps 도달: 도구 {tool_calls}")

    return AgentResult(
        text=text, steps=steps, truncated=truncated, tool_calls=tool_calls
    )
