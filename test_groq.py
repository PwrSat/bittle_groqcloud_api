from groq import Groq

client = Groq(api_key="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")

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Halo, kamu siapa?"}]
)

print(response.choices[0].message.content)