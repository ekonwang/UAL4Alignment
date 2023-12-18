import openai

def get_gpt_response(params, messages=None, temperature=None):
    resp = openai.ChatCompletion.create(
        model=params.model_name,
        messages=messages,
        temperature=params.temperature if temperature is None else temperature,
        max_tokens=params.max_tokens,
        top_p=params.top_p,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return resp["choices"][0]["message"]["content"]