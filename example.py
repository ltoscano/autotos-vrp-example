import math
from typing import List, Tuple
from openai import OpenAI
import re

OPENAI_API_KEY = "your-api-key"

def extract_python_code(text: str) -> str:
    """
    Extracts Python code from a given text string that contains code blocks.
    
    Args:
    - text: A string that may contain Python code blocks enclosed in triple backticks.
    
    Returns:
    - A string containing all the Python code extracted from the code blocks. If no code blocks are found, returns the original text.
    """
    # Regex pattern to match Python code blocks
    pattern = r'```python\s*([\s\S]*?)\s*```'
    
    # Find all matches for the pattern
    matches = re.findall(pattern, text)
    
    # If matches are found, join all extracted code blocks into a single string
    # Otherwise, return the original text
    if matches:
        extracted_code = '\n\n'.join(matches)
        return extracted_code
    else:
        return text

# Data type to represent a point (x, y)
Point = Tuple[int, int]
# Data type to represent a route (list of points)
Route = List[Point]
# Data type to represent a state (list of routes)
State = List[Route]

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=OPENAI_API_KEY)
# Set the model to be used for generating code
model = 'gpt-4o-mini'

def distance(p1: Point, p2: Point) -> float:
    """
    Calculates the Euclidean distance between two points.
    
    Args:
    - p1: A tuple representing the first point (x1, y1).
    - p2: A tuple representing the second point (x2, y2).
    
    Returns:
    - The Euclidean distance between the two points.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_expected_successors(state: State) -> int:
    """
    Calculates the expected number of successor states for a given state.
    
    Args:
    - state: A list of routes, where each route is a list of points.
    
    Returns:
    - The expected number of successor states that can be generated from the given state.
    """
    num_routes = len(state)
    # Calculate the number of internal swaps within each route
    internal_swaps = sum((len(route) - 2) * (len(route) - 3) // 2 for route in state)
    # Calculate the number of swaps between different routes
    inter_route_swaps = sum((len(state[i]) - 2) * (len(state[j]) - 2) 
                            for i in range(num_routes) 
                            for j in range(i+1, num_routes))
    return internal_swaps + inter_route_swaps

def get_llm_functions(feedback: str = "") -> str:
    """
    Generates Python functions for the Vehicle Routing Problem (VRP) using a language model.
    
    Args:
    - feedback: A string containing feedback from the previous implementation, if any.
    
    Returns:
    - A string containing the Python code for the 'goal' and 'successor' functions.
    """
    base_prompt = """
    Implement two Python functions for the Vehicle Routing Problem (VRP):

    1. A 'goal' function that calculates the total cost of a state.
    2. A 'successor' function that generates all possible successor states.

    Use these definitions:

    - Point = Tuple[int, int]  # A point (x, y)
    - Route = List[Point]      # A route (list of points)
    - State = List[Route]      # A state (list of routes)

    The distance function is already provided:

    def distance(p1: Point, p2: Point) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    Implement these functions:

    1. def goal(state: State) -> float:
       This function should calculate and return the total cost of all routes in the state.
       - For each route, sum the distances between each pair of consecutive points.
       - Sum the costs of all routes.

    2. def successor(state: State) -> List[State]:
       This function should generate and return all possible successor states.
       - Generate new states by swapping pairs of customers within the same route.
       - Generate new states by swapping pairs of customers between different routes.
       - Never swap the depot (0,0), which must always be the first and last point of each route.
       - All successors must be valid and unique states, without duplicates.

    Example state:
    state = [
        [(0,0), (1,5), (2,3), (0,0)],  # Route 1
        [(0,0), (5,1), (3,2), (0,0)]   # Route 2
    ]

    Provide only the function code, without additional explanations.
    """

    # If feedback is provided, include it in the prompt for the language model
    if feedback:
        prompt = f"{base_prompt}\n\nFeedback from the last implementation:\n{feedback}\n\nPlease correct the implementation considering this feedback."
    else:
        prompt = base_prompt

    # Generate the code using the language model
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the Python code from the response
    return extract_python_code(response.choices[0].message.content)

def visualize_states(states: List[State], limit: int = 5):
    """
    Prints a visualization of a list of states, up to a specified limit.
    
    Args:
    - states: A list of states to visualize.
    - limit: The maximum number of states to visualize.
    """
    print(f"\nVisualizing up to {limit} states:")
    for i, state in enumerate(states[:limit]):
        print(f"\nState {i + 1}:")
        for j, route in enumerate(state):
            print(f"  Route {j + 1}: {route}")
    if len(states) > limit:
        print(f"\n... and {len(states) - limit} more states")

def test_vrp_functions(goal: callable, successor: callable) -> str:
    """
    Tests the 'goal' and 'successor' functions for the Vehicle Routing Problem (VRP).
    
    Args:
    - goal: The function to calculate the total cost of a state.
    - successor: The function to generate successor states.
    
    Returns:
    - An empty string if all tests pass, or an error message indicating the failed test.
    """
    test_state = [
        [(0,0), (1,5), (2,3), (0,0)],  # Route 1
        [(0,0), (5,1), (3,2), (0,0)]   # Route 2
    ]
    
    # Test the 'goal' function
    expected_cost = (
        distance((0,0), (1,5)) + distance((1,5), (2,3)) + distance((2,3), (0,0)) +
        distance((0,0), (5,1)) + distance((5,1), (3,2)) + distance((3,2), (0,0))
    )
    calculated_cost = goal(test_state)
    print(f"Expected cost: {expected_cost}")
    print(f"Calculated cost: {calculated_cost}")
    if not math.isclose(expected_cost, calculated_cost):
        return "Goal function test failed"
    
    # Test the 'successor' function
    successors = successor(test_state)
    expected_num_successors = calculate_expected_successors(test_state)
    print(f"Number of successors: {len(successors)}")
    if len(successors) != expected_num_successors:
        return f"Incorrect number of successors. Expected {expected_num_successors}, got {len(successors)}"
    
    # Verify that all successors are unique
    unique_successors = set(tuple(tuple(tuple(p) for p in route) for route in state) for state in successors)
    if len(unique_successors) != expected_num_successors:
        return f"Incorrect number of unique successors. Expected {expected_num_successors}, got {len(unique_successors)}"
    
    # Verify that the depot (0,0) is always the first and last point of each route
    for state in successors:
        for route in state:
            if route[0] != (0,0) or route[-1] != (0,0):
                return "Depot constraint violated"
    
    visualize_states(successors)
    
    print("All tests passed successfully!")
    
    return ""

def autotos():
    """
    Automatically generates and tests 'goal' and 'successor' functions for the VRP using a language model.
    """
    max_attempts = 5
    attempt = 0
    feedback = ""

    while attempt < max_attempts:
        print(f"\nAttempt {attempt + 1}:")
        llm_functions = get_llm_functions(feedback)

        # Execute the code generated by the language model
        try:
            exec(llm_functions, globals())
        except Exception as e:
            feedback = f"Error in code execution: {str(e)}"
            print(feedback)
            attempt += 1
            continue

        # Test the functions
        feedback = test_vrp_functions(goal, successor)
        
        if not feedback:
            print("Success! All tests passed.")
            break
        else:
            print(f"Test failed.\nPrevious wrong functions:\n{llm_functions}\nFeedback: {feedback}")
            attempt += 1

    if attempt == max_attempts:
        print("Maximum attempts reached. Could not generate correct functions.")

if __name__ == "__main__":
    autotos()