"""Minimal OpenAI-compatible chat client over urllib (D-011). No SDK dependency.

Works with OpenAI, Ollama, Groq, OpenRouter, vLLM, LM Studio: anything that
serves POST {base_url}/chat/completions. Provider, URL, model and the name of
the key's environment variable come from config.llm_settings().
"""

import json
import time
import urllib.error
import urllib.request

from ragbot import config


class LLMError(Exception):
    """The LLM call failed (network, HTTP error, or an unexpected response shape)."""


class LLMNotConfigured(LLMError):
    """The chosen provider needs an API key that is not set in the environment."""


def _http_post(url, headers, body, timeout):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class ChatClient:
    def __init__(self, provider=None, post=_http_post, sleep=time.sleep):
        self.settings = config.llm_settings(provider)
        self.post, self.sleep = post, sleep  # injectable so tests need no network
        key_env = self.settings["key_env"]
        self.api_key = config.llm_api_key(provider)
        if key_env and not self.api_key:
            raise LLMNotConfigured(f"set {key_env} (provider {self.settings['provider']!r}), or choose a keyless "
                                   f"provider such as ollama with RAGBOT_LLM_PROVIDER")

    @property
    def model(self):
        return self.settings["model"]

    def chat(self, messages, max_tokens, temperature=config.TEMPERATURE):
        """Send chat messages, return the assistant text. Retries 429/5xx and network errors."""
        url = f"{self.settings['base_url']}/chat/completions"
        headers = {"Content-Type": "application/json", "User-Agent": "rag-chatbot/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"  # never logged or printed
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": temperature, "max_tokens": max_tokens}).encode()
        last = None
        for attempt in range(config.LLM_MAX_RETRIES + 1):
            try:
                data = json.loads(self.post(url, headers, body, config.LLM_TIMEOUT_S))
                return data["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code not in (429, 500, 502, 503, 504):
                    raise LLMError(f"{last} from {self.settings['provider']}: {_error_text(e)}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = f"network error: {getattr(e, 'reason', e)}"
            except (KeyError, IndexError, TypeError, ValueError) as e:
                raise LLMError(f"unexpected response shape from {self.settings['provider']}: {e!r}") from e
            if attempt < config.LLM_MAX_RETRIES:
                self.sleep(2 ** (attempt + 1))
        raise LLMError(f"{last} after {config.LLM_MAX_RETRIES + 1} attempts ({self.settings['base_url']})")


def _error_text(e):
    try:
        return json.loads(e.read())["error"]["message"][:300]
    except Exception:  # error bodies differ between providers; never fail while reporting a failure
        return e.reason
