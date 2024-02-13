from langchain_openai import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI()

code_prompt = PromptTemplate.from_template(
    template = "Write a simple {language} function to {task}.",
)
code_chain = LLMChain(
    llm = llm,
    prompt = code_prompt,
    output_key = "code",
)

test_prompt = PromptTemplate.from_template(
    template = "Write a test for the following {language} code:\n{code}",
)
test_chain = LLMChain(
    llm = llm,
    prompt = test_prompt,
    output_key = "test",
)

chain = SequentialChain(
    chains = [code_chain, test_chain],
    input_variables = ["language", "task"],
    output_variables = ["test", "code"],
)

result = chain.invoke(
    input = {
        "language": "python",
        "task": "return an array from 1 to 10"
    }
)
print(result)


