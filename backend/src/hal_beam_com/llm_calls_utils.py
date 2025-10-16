def call_o3_mini(prompt):

    client = AzureOpenAI(
            api_key = os.environ.get('AZURE_o1_API_KEY'),  
            api_version='2024-12-01-preview',
            azure_endpoint = os.environ.get('AZURE_o1_API_BASE'),
            azure_deployment = os.environ.get('AZURE_o3_mini_DEPLOYMENT')
            )
            
    response = client.chat.completions.create(
        model="o3-mini", # model = "deployment_name".
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print(response.choices[0].message.content)

    return response.choices[0].message.content


def execute_llm_call(model, prompt, temperature=0.7):
    """
    Executes an LLM call with standardized provider handling.
    
    Args:
        llm: The loaded language model
        prompt_template: The formatted prompt template
        user_prompt: The user's input text
        strip_markdown: Whether to strip markdown formatting from output
        
    Returns:
        str: The processed LLM output
    """
    print("in Ollama")
    llm = Ollama(
    model=model,  # or any other model you have pulled
    temperature=temperature
)
    response = llm(prompt)
    
    return response