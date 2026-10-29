"""OpenAI Responses API 호출 공통 코드 (Structured Outputs).

이전 개인 프로젝트의 호출·오류 처리 방식을 가져왔다. 표준 라이브러리만 사용한다.
키와 모델은 환경변수로만 받는다: OPENAI_API_KEY, OPENAI_MODEL
"""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = "https://api.openai.com/v1/responses"


class LLMError(RuntimeError):
    pass


def build_request(model, instructions, user_content, schema, schema_name, max_output_tokens=8000):
    return {
        "model": model,
        "store": False,
        "max_output_tokens": max_output_tokens,
        "instructions": instructions,
        "input": [{"role": "user", "content": user_content}],
        "text": {"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}},
    }


def parse_response(response):
    """API 응답에서 JSON 결과를 꺼낸다. 미완료·거절·잘못된 JSON은 오류."""
    if not isinstance(response, dict) or response.get("status") != "completed":
        reason = (response or {}).get("incomplete_details") if isinstance(response, dict) else None
        raise LLMError(f"API 응답이 완료되지 않았습니다 ({reason or '상태 확인 필요'}). 출력 제한을 늘리거나 묶음 크기를 줄이세요.")
    chunks = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "refusal":
                raise LLMError("모델이 답변을 거절했습니다.")
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    try:
        return json.loads("".join(chunks))
    except (ValueError, TypeError):
        raise LLMError("API에서 유효한 JSON을 받지 못했습니다.") from None


def call(request, timeout=120):
    """요청을 보내고 JSON 결과를 돌려준다. 자동 재시도는 하지 않는다."""
    if not request.get("model"):
        raise LLMError("--model 또는 OPENAI_MODEL에 Structured Outputs를 지원하는 모델 ID를 지정하세요.")
    return parse_response(post(ENDPOINT, request, timeout))


EMBEDDING_ENDPOINT = "https://api.openai.com/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def embed(texts, model=None, timeout=120):
    """문장 목록 → 임베딩 벡터 목록. 모델: OPENAI_EMBEDDING_MODEL 또는 text-embedding-3-small."""
    model = model or os.environ.get("OPENAI_EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
    response = post(EMBEDDING_ENDPOINT, {"model": model, "input": texts}, timeout)
    data = sorted(response.get("data", []), key=lambda item: item["index"])
    if len(data) != len(texts):
        raise LLMError("임베딩 개수가 입력 개수와 다릅니다.")
    return [item["embedding"] for item in data]


def post(url, payload, timeout=120):
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise LLMError("OPENAI_API_KEY 환경변수가 필요합니다. 키 없이 확인하려면 --dry-run을 쓰세요.")
    http_request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=timeout) as result:
            response = json.loads(result.read().decode("utf-8"))
    except HTTPError as error:
        hints = {
            400: "모델이 Responses API와 Structured Outputs를 지원하는지 확인하세요.",
            401: "API 키를 확인하세요.",
            403: "모델·프로젝트 접근 권한을 확인하세요.",
            404: "모델 ID를 확인하세요.",
            429: "사용량 한도나 요청 빈도를 확인하세요 (크레딧 잔액 포함).",
        }
        raise LLMError(f"API 요청 실패 (HTTP {error.code}). {hints.get(error.code, '잠시 후 다시 시도하세요.')}") from None
    except (URLError, TimeoutError, OSError):
        raise LLMError("API 연결 실패 또는 시간 초과입니다. 네트워크를 확인하세요.") from None
    except (ValueError, UnicodeError):
        raise LLMError("API 응답을 읽을 수 없습니다.") from None
    return response


def model_from_env(model=None):
    return (model or os.environ.get("OPENAI_MODEL", "")).strip()
