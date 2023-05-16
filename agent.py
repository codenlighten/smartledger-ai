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

def getAIResponse(text, context, purpose):
    embeddings, index, chunks = load_embeddings_and_index("./embeddings/new/embeddings.json")
    top_result = vector_search(text, index, embeddings, chunks)
    print(top_result)
    top_result = summarize(top_result, purpose)
    prompt = f"You are an AI Agent. Your purpose is: {purpose}. Consider the following user request: {text}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": top_result},
            {"role": "user", "content": context},
        ]
    )
    res = response['choices'][0]['message']['content']
    print('AI: ')
    print(res)
    return res

def clarify(user_request):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": "You are a purpose clarification AI. Please clarify the user's purpose based on their request."},
            {"role": "user", "content": user_request},
        ]
    )
    purpose = response['choices'][0]['message']['content']
    print('Clarified purpose:')
    print(purpose)
    return purpose

def summarize(data, purpose):
    prompt = f"Please summarize the following information while retaining any important details related to our purpose: {purpose}.\n\n{data}"
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

def define_task_list(clarifiedPrompt, purpose):
    role_context = f'You are an AI task definition engineer. Your purpose is: {purpose}. Create tasks to accomplish this purpose. The tasks should involve using the AI and accessing the local filesystem to keep track of user purpose, tasks to be executed, and roles of agents executing the task.'
    task_list = getAIResponse(clarifiedPrompt, role_context, purpose)
    return task_list

def create_roles(task_list, purpose):
    role_context = f'You are an AI role creation engineer. Your purpose is: {purpose}. Create roles for the AI and filesystem functions to accomplish the following task list. Assign a function for each task and assign the task to a role.'
    role_list = getAIResponse(task_list, role_context, purpose)
    return role_list

def delegate_tasks(roles, purpose):
    delegate_context = f'You are a task and function delegation engineer. Your purpose is: {purpose}. Provide an ordered list of roles and the code functions the AI role will execute to accomplish our purpose.'
    return getAIResponse(roles, delegate_context, purpose)

def execute_tasks(tasks_to_execute, purpose):
    execution_context = f"You are a task, function, and role execution engineer. Your purpose is: {purpose}. Please generate code based upon the users request."
    tasks = getAIResponse(tasks_to_execute, execution_context, purpose)
    with open('tasks.js', 'w') as f:
        for i, task in enumerate(tasks.split('\n'), 1):
            f.write(f'// Task {i}\n{task}\n\n')
    return tasks

def iterate(executed_tasks, purpose):
    prompt = f"You are an AI iteration engineer. Your purpose is: {purpose}. Review the user's request and the results of our agent's actions. Make any modifications to better accomplish our user purpose."
    return getAIResponse(executed_tasks, prompt, purpose)

def main_agent(user_request, api_key, satisfied):
    global state
    openai.api_key = api_key
    state["purpose"] = clarify(user_request)
    logging.info("Entering task list")
    state["task_list"] = define_task_list(state["purpose"], state["purpose"])
    logging.info("Entering role list")
    state["role_list"] = create_roles(state["task_list"], state["purpose"])
    logging.info("Entering delegation list")
    tasks_to_execute = delegate_tasks(state["role_list"], state["purpose"])
    while True:
        executed_tasks = execute_tasks(tasks_to_execute, state["purpose"])
        iterations = iterate(executed_tasks, state["purpose"])
        logging.info(iterations)
        # Check if user is satisfied
        if satisfied.lower() not in ["yes", "no"]:
            logging.error("Invalid response. Please answer with 'yes' or 'no'.")
            continue
        if satisfied.lower() == "yes":
            break
    logging.info(executed_tasks)
    return executed_tasks


while True:
    openai.api_key = input('add OPENAI API KEY: ')
    user_request = input('Please provide request (type "exit" to quit): ')
    if user_request.lower() == 'exit':
        break
    if user_request.strip() == "":
        logging.error("User request cannot be empty.")
        continue
    main_agent(user_request)