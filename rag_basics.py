knowledge = """
The Transformer is a neural network architecture introduced in
the paper Attention Is All You Need.

It relies primarily on attention mechanisms rather than recurrence.

The original Transformer architecture contains an encoder and
a decoder.
"""
question = "What is the architecture of GPT-3?"
retrieved_context = knowledge
prompt = f"""
Answer the question using the provided context.

Context:
{retrieved_context}

Question:
{question}
"""
print(prompt)