"""Increase DS response allowance without replacing an active controller.

Executed at a CPython pending-call boundary. Existing requests are untouched.
No search code, prompt, temperature, reasoning effort or task budget changes.
"""
import sys


def install(record):
    backend = sys.modules['llm.openai']
    if getattr(backend.OpenAI, '_mle_response_cap_fixed', False):
        return
    previous_factory = backend.OpenAI

    def factory(*args, **kwargs):
        client = previous_factory(*args, **kwargs)
        previous_create = client.chat.completions.create

        def create(**params):
            if params.get('model') == 'deepseek-flash':
                old_cap = params.get('max_tokens')
                params['max_tokens'] = max(int(old_cap or 0), 65536)
                record(event='output_cap_request', previous_max_tokens=old_cap,
                       max_tokens=params['max_tokens'], streaming=bool(params.get('stream')))
            result = previous_create(**params)
            if not params.get('stream'):
                record(event='output_cap_response', max_tokens=params.get('max_tokens'),
                       finish_reasons=[choice.finish_reason for choice in result.choices])
                return result

            def chunks():
                for chunk in result:
                    reasons=[choice.finish_reason for choice in chunk.choices if choice.finish_reason]
                    if reasons:
                        record(event='output_cap_response', max_tokens=params.get('max_tokens'),
                               finish_reasons=reasons)
                    yield chunk
            return chunks()

        client.chat.completions.create = create
        return client

    factory._mle_response_cap_fixed = True
    backend.OpenAI = factory
    record(event='output_cap_live_fix_installed', max_tokens=65536,
           existing_requests_unchanged=True, search_source_unchanged=True)


install(record)
