import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import pickle as pkl
load_dotenv()

## Langsmith params for observability
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_PROJECT'] = 'LLM_OBS_YT'
os.environ['LANGSMITH_TRACING']="true"

### import the Agent
from research import AppAgent
# Initialize the agent
agent = AppAgent()


prompts = ['Tell me about mutlihead attention in transformers',
             'who is the winner of Last T20 Cricket World Cup?']

print('Question:',prompts)


expected_tool_calls = [['transfer_to_rag_expert', 'transfer_back_to_supervisor'],
                       ['transfer_to_research_expert', 'transfer_back_to_supervisor']]

print('Expected Tool Calls:', expected_tool_calls)

def extract_tool_calls(messages: List[Any]) -> List[str]:
    """Extract tool call names from messages, safely handling messages without tool_calls."""
    tool_call_names = []
    for message in messages:
        # Check if message is a dict and has tool_calls
        if isinstance(message, dict) and message.get("tool_calls"):
            tool_call_names.extend([call["name"].lower() for call in message["tool_calls"]])
        # Check if message is an object with tool_calls attribute
        elif hasattr(message, "tool_calls") and message.tool_calls:
            tool_call_names.extend([call["name"].lower() for call in message.tool_calls])
    
    return tool_call_names


import pytest
from langsmith import testing as t

@pytest.mark.langsmith
@pytest.mark.parametrize(
    "prompts, expected_tool_calls",
    [   # Pick some examples with e-mail reply expected
        (prompts[0],expected_tool_calls[0]),
        (prompts[1],expected_tool_calls[1]),
    ],
)
def test_agent_tool_calls(prompts, expected_tool_calls):

    """Test that the agent calls the expected tools."""
    # Initialize the agent
    result = agent.invoke(prompts)
    # Extract tool calls from the agent's response
    executed_tool_calls = extract_tool_calls(result['messages'])
                        
    # Check if all expected tool calls are in the extracted ones
    missing_calls = [call for call in expected_tool_calls if call.lower() not in executed_tool_calls]
    
    t.log_outputs({
                "missing_calls": missing_calls,
                "executed_tool_calls": executed_tool_calls,
                "expected_tool_calls": expected_tool_calls
            })

    # Test passes if no expected calls are missing
    assert len(missing_calls) == 0


## run in the comman line
## ! LANGSMITH_TEST_SUITE='Research Agent Tools Calling test'  pytest test_agent_toolcalling.py