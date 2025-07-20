import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from langsmith import expect
import pytest
import warnings
warnings.filterwarnings('ignore')

## Langsmith params for observability
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_PROJECT'] = 'LLM_OBS_YT'
os.environ['LANGSMITH_TRACING']="true"

### import the Agent
from research import AppAgent
# Initialize the agent
agent = AppAgent()

dataset = pd.read_csv('agent_qna.csv')

prompts = dataset['Questions/Prompt'].tolist()
references = dataset['Answer'].tolist()


@pytest.mark.langsmith(output_keys=["expectation"])
@pytest.mark.parametrize(
    "prompts, references",
    [
       (prompts[0], references[0]),
       (prompts[1], references[1]),
       (prompts[2], references[2]),
       (prompts[3], references[3]),
       (prompts[4], references[4]),
       (prompts[5], references[5]),
       (prompts[6], references[6]),
       (prompts[7], references[7]),
       (prompts[8], references[8]),
       (prompts[9], references[9]),
    ],
)
def test_embedding_similarity(prompts, references):
    response = agent.invoke(prompts)
    prediction = response['messages'][-1].content
    expect.embedding_distance(
        # This step logs the distance as feedback for this run
        prediction=prediction, reference=references,
    # logs 'expectation' feedback
    ).to_be_less_than(0.5) # Optional predicate to assert against