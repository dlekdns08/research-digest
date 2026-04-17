"""LLM summarization using the Anthropic SDK with prompt caching."""

from __future__ import annotations

from anthropic import Anthropic

_SYSTEM_PROMPT = """당신은 AI/ML 연구 논문을 빠르게 요약해주는 한국어 리서치 어시스턴트입니다.

사용자가 제공하는 논문의 제목(title)과 초록(abstract)을 읽고 다음 형식으로 한국어 요약을 작성하세요.

형식:
요약: <2-3 문장으로 핵심 아이디어와 결과를 설명>
왜 볼만한지: <한 줄로 연구자/엔지니어가 이 논문을 주목해야 하는 이유>

규칙:
- 한국어로 답하고 전문 용어는 원어를 병기해도 좋습니다.
- 추측하지 말고 초록에 명시된 내용만 사용하세요.
- 불필요한 서론/꼬리말 없이 바로 형식대로 출력하세요.
"""


def summarize_paper(
    title: str,
    abstract: str,
    model: str,
    api_key: str | None = None,
    max_tokens: int = 400,
) -> str:
    """Return a short Korean summary + 'why it's worth reading' line.

    Uses prompt caching on the system message so repeated calls in the
    same run hit the cache after the first one.
    """
    client = Anthropic(api_key=api_key) if api_key else Anthropic()

    user_content = f"제목(title): {title}\n\n초록(abstract):\n{abstract.strip()}"

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )

    parts = [blk.text for blk in resp.content if getattr(blk, "type", None) == "text"]
    return "\n".join(parts).strip()
