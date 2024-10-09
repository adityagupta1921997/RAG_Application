RAG_PROMPT_TEMPLATE = """
You are given user review and the context from the product manual.
User review may not always be correct, eg: he may mention in the review that product is having 2 years of warranty but context may contain warranty period as 5 years.
Assume information present in the context is always correct.
So you have to find relevant information as per the user review from the given context and return back the correct information, so that later I can compare the information mentioned in user review with the information given by you.
Remember your source of information is just the context below.
Give your answer in brief.
<context>
{context}
</context>

Review: {question}
"""