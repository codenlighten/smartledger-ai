import os
import openai
import logging
from main import load_embeddings_and_index, vector_search

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Global state
state = {
    "api_key": None,
    "purpose": "",
    "role_list": [],
    "task_list": [],
}


def getAIResponse(text, context):
    try:
        # Load embeddings and perform vector search
        embeddings, index, chunks = load_embeddings_and_index("./embeddings/new/embeddings.json")
        top_result = vector_search(text, index, embeddings, chunks)
        top_result = summarize(top_result)
        # Use the top result as additional context for the GPT-3 completion
        prompt = f"Consider the following context:\n{context}\n and the following request\n {text}\n\nConsider the following related item: {top_result}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["###"]
        )
        logging.info('AI Agent:')
        logging.info(response.choices[0].text.strip())
        return response.choices[0].text.strip()
    except openai.Error as e:
        logging.error(f"OpenAI error: {e}")
        return "An error occurred while processing your request."
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return "An error occurred while processing your request."


def clarify(user_request):
    global purpose
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=4096 - len(user_request),
        messages=[
            {"role": "system",
             "content": "You are a master task and purpose engineer. Please provide a main purpose for user's request to be used in by a task engineer to decide the tasks to accomplish the purpose of the user:"},
            {"role": "user", "content": user_request},
        ]
    )
    purpose = response['choices'][0]['message']['content']
    print('I am clarifying your purpose:')
    print(purpose)
    return purpose



def summarize(data):
    try:
        # Load embeddings and perform vector search
        prompt = f"Please summarize the following {purpose}:\n{data}\n any important code or necessary information to accomplish our task in the future"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["###"]
        )
        logging.info('AI Agent Summarizing:')
        logging.info(response.choices[0].text.strip())
        return response.choices[0].text.strip()
    except openai.Error as e:
        logging.error(f"OpenAI error: {e}")
        return "An error occurred while processing your request."
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return "An error occurred while processing your request."



def define_task_list(clarifiedPrompt):
    global task_list
    role_context = 'You are a task defintions engineer for this project. You are in charge an chatgpt prompt that will create tasks to accomplish the following purpose. The tasks will be an order of functions that will use or chatgpt and access to the local filesystem to keep track of user purpose, tasks to be executed, and roles of agents executing the task. We have these roles you can assign tasks to: Provide a list of only the roles and their functions to execute'
    task_list = getAIResponse(clarifiedPrompt, role_context)
    return task_list


def create_roles(task_list):
    global role_list
    role_context = 'You are a role creation engineer for this project. You will create roles for ai and file system functions to accomplish the following task list. Assign a function for each task and assign the task to a role. Provide a list of only roles and tasks to be executed'
    role_list = getAIResponse(task_list, role_context)
    return role_list


def delegate_tasks(roles):
    delegate_context = 'You are an task and function delegation engineer for this project. You will clarify anything needed to accomplish or purpose and create an order of role functions to be executed to accomplish our purpose. Provide an ordered list of functions and roles to accomplish our purpose'
    return getAIResponse(roles, delegate_context)


def execute_tasks(tasks_to_execute):
    execution_context = "You are a task, function, and role execution engineer for this project. You will examine the users purpose, the task list, the roles, and the order of roles. Please generate JavaScript code for a transferable, fungible token using the bsv@1.5 library. This code should include the creation, transfer, and validation of the token."
    tasks = getAIResponse(tasks_to_execute, execution_context)
    with open('tasks.js', 'w') as f:
        for i, task in enumerate(tasks.split('\n'), 1):
            f.write(f'// Task {i}\n{task}\n\n')
    return tasks


def iterate(executed_tasks):
    iteration_context = 'You are an iteration engineer for this project. You will iterate over our task list with our user input if required  to get the code or purpose completed correctly for the user. We will use gpt as the tool for each function. Provide a list of roles'
    print(f"Iterating over: {executed_tasks}")  # Debugging line
    return getAIResponse(executed_tasks, iteration_context)


def main_agent(user_request):
    global state
    state["purpose"] = clarify(user_request)
    state["task_list"] = define_task_list(state["purpose"])
    state["role_list"] = create_roles(state["task_list"])
    tasks_to_execute = delegate_tasks(state["role_list"])
    while True:
        executed_tasks = execute_tasks(tasks_to_execute)
        # summarize_data = summarize(executed_tasks)
        iterations = iterate(executed_tasks)
        print(iterations)
        # Ask user if they are satisfied
        user_satisfied = input("Are you satisfied with the results? (yes/no): ")
        if user_satisfied.lower() == "yes":
            break
    print(executed_tasks)


state["api_key"] = input('Enter your OpenAI API key: ')
openai.api_key = state["api_key"]

while True:
    user_request = input('Please provide request (type "exit" to quit): ')
    if user_request.lower() == 'exit':
        break
    main_agent(user_request)
