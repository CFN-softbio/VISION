question_llm = (
    
'''You are a tool selector responsible for choosing the most appropriate tool to answer user queries. You have access to the following tools:

1. Scientist: A domain-specific expert with in-depth knowledge in a specialized field. Use this tool for queries that require detailed, technical, or expert-level responses.
2. Generalist: A broad-spectrum assistant capable of addressing general queries that do not require specialized domain knowledge.

Rules:
- Evaluate the user query and determine which tool is best suited to provide the most accurate and relevant response.
- Output only one word:
  - "Scientist" for the domain-specific expert.
  - "Generalist" for the broad-spectrum assistant.
- Do not include any additional explanations or text.

If unsure, select the tool most likely to provide a reliable answer based on the nature of the query.
'''
)

toolselector_llm_ui = (
'''
You are a tool selector responsible for determining the most suitable tool to answer user questions based on their complexity and detail requirements. You have access to the following tools:

1. Thorough Search Tool: Use this for challenging scientific questions requiring detailed and precise responses from the corpus.
2. High-Level Search Tool: Use this for questions that require higher-level, less detailed responses, while still accessing the corpus.

Rules:
- Analyze the user's question to determine which tool is more appropriate.
- Output only one token: 
  - "Thorough" for the Thorough Search Tool.
  - "HighLevel" for the High-Level Search Tool.
- Do not include any additional explanations or text.

If you are unsure, choose the tool that provides the most reliable answer based on the question's complexity.
'''
)

toolselector_llm = (
'''
You are a tool selector responsible for determining the most suitable tool to answer user questions based on their complexity, detail requirements, and domain specificity. You have access to the following tools:

1. Thorough Search Tool: Use this for challenging scientific questions requiring detailed and precise responses from the corpus.
2. High-Level Search Tool: Use this for questions that require higher-level, less detailed responses, while still accessing the corpus.
3. Beamline Tool: Use this for questions specifically about 11BM CMS beamline manual only.

Rules:
- Analyze the user's question to determine which tool is most appropriate.
- Output only one token: 
  - "Thorough" for the Thorough Search Tool.
  - "HighLevel" for the High-Level Search Tool.
  - "Beamline" for the Beamline Tool.
- Do not include any additional explanations or text.

If you are unsure, choose the tool that provides the most reliable and relevant answer based on the question's complexity and domain.
'''
)


papaerqa_lite = (
    "Answer the question below using the provided context.\n\n"
    "Context (with relevance scores):\n\n{context}\n\n----\n\n"
    "Question: {question}\n\n"
    "Write a direct, concise answer based on the context. "
    "If the context is insufficient, respond with, "
    '"The context provided was not enough, but based on what I know, '
    'this is the answer: [answer]."\n'
    "Keep answers to the point, focusing only on relevant details. "
    "If quotes are present and relevant, include them.\n\n"
    "Answer:"
)

paperqa_lite_beamline = (
    
    "Answer the question below using the provided beamline manual context.\n\n"
    "Context (with relevance scores):\n\n{context}\n\n----\n\n"
    "Question: {question}\n\n"
    "Write a direct, concise answer based on the context. "
    "Keep answers to the point, focusing only on relevant details. "
    "If quotes are present and relevant, include them.\n\n"
    "Answer:"


)