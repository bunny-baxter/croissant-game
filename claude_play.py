import argparse
import datetime
import json
import os

import anthropic

import fake_anthropic_client
import game_model

parser = argparse.ArgumentParser(prog = "Claude plays the croissant game via the Anthropic API.")
parser.add_argument("--real_api", action = "store_true", help = "Use the real API. If not enabled, uses a local fake instead.")
parser.add_argument("--thoughts", action = "store_true", help = "Prompt model to reason about their action before taking it.")
parser.add_argument("--fake_reply_text", help = "If using the local fake API, use this string as the reply.")
args = parser.parse_args()

assert not args.real_api or not args.fake_reply_text

TEMPERATURE = 0.8

def value_object_to_dict(obj):
    result = {}
    for k, v in obj.__dict__.items():
        if type(v) == list:
            v_list = [ value_object_to_dict(e) for e in v ]
            result[k] = v_list
        elif hasattr(v, "__dict__"):
            v_dict = value_object_to_dict(v)
            result[k] = v_dict
        else:
            result[k] = v
    return result

if args.real_api:
    client = anthropic.Anthropic()
else:
    reply_text = args.fake_reply_text or "Labor"
    client = fake_anthropic_client.FakeAnthropicClient(reply_text = reply_text)

system_prompt = f"""\
You are an expert strategist playing a game called The Croissant Game. Your goal is to get the best score possible.

The game will last for 12 turns. You start with 0 dollars. You get 1 point for every croissant you buy.

On every turn, you can execute one of the following actions:
* The "Labor" action gives you {game_model.config["labor_payoff"]} dollars.
* The "Invest" action costs {game_model.config["invest_cost"]} dollars, and will give you {game_model.config["invest_payoff"]} \
dollars {game_model.config["invest_lag_turns"]} turns later.
* The "Consume 1" action buys 1 croissant for 1 dollar.
* The "Consume 5" action buys 5 croissants for 5 dollars.
* The "Consume 20" action buys 20 croissants for 20 dollars.\
"""

game = game_model.CroissantGame()
message_log = []

def preprocess_message_log_for_api():
    result = []
    for message in message_log:
        # Removes other elements from the message dict, which are not allowed by the API.
        result.append({ "role": message["role"], "content": message["content"] })
    return result

def send_user_message(prompt):
    message_log.append({
        "role": "user",
        "content": [
            { "type": "text", "text": prompt }
        ]
    })
    reply = client.messages.create(
        model = "claude-3-5-sonnet-20241022",
        max_tokens = 1000,
        temperature = TEMPERATURE,
        system = system_prompt,
        messages = preprocess_message_log_for_api()
    )
    if reply.type != "message" or len(reply.content) != 1:
        print(f"Unexpected server reply: {reply}")
        raise Exception()
    message_log.append(value_object_to_dict(reply))
    return reply.content[0].text

print("SYSTEM PROMPT:")
print(system_prompt)
print()

def send_prompt_and_print_conversation(prompt):
    print("USER:")
    print(prompt)
    reply_text = send_user_message(prompt)
    print("MODEL:")
    print(reply_text)
    return reply_text

send_prompt_and_print_conversation("What will your strategy be?")
print()

while game.turns_left > 0:
    try:
        if args.thoughts:
            thoughts_prompt = f"There are {game.turns_left} turns remaining. You have {game.money} dollars and have {game.croissants} croissants. Given your strategy, think about what you should do this turn."
            send_prompt_and_print_conversation(thoughts_prompt)
            action_prompt = "Given your thinking for this turn, what is your action? Respond only with the action."
            reply_text = send_prompt_and_print_conversation(action_prompt)
        else:
            prompt = f"There are {game.turns_left} turns remaining. You have {game.money} dollars and have {game.croissants} croissants. Given your strategy, what is your action? Respond only with the action."
            reply_text = send_prompt_and_print_conversation(prompt)
        print()
    except Exception as e:
        break

    reply_text_lower = reply_text.lower()
    try:
        if "labor" in reply_text_lower:
            game.execute_labor()
        elif "invest" in reply_text_lower:
            game.execute_invest()
        elif "consume" in reply_text_lower:
            if "1" in reply_text_lower:
                game.execute_consume(1)
            elif "5" in reply_text_lower:
                game.execute_consume(5)
            elif "20" in reply_text_lower:
                game.execute_consume(20)
            else:
                print("Model reply did not contain a valid Consume argument.\n")
                break
        else:
            print("Model reply did not contain a valid action.\n")
            break
    except game_model.InvalidActionException as e:
        print(f"Model attempted invalid action: {e}\n")
        break

print(f"Score was {game.croissants} croissants.")

json_str = json.dumps({
    "temperature": TEMPERATURE,
    "system": system_prompt,
    "messages": message_log,
    "final_money": game.money,
    "final_score": game.croissants,
}, indent = 2)

date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if args.real_api:
    thoughts = "nothoughts"
    if args.thoughts:
        thoughts = "yesthoughts"
    log_filename = f"logs/claude_{thoughts}_{date_str}.json"
else:
    log_filename = f"logs/fake_{date_str}.json"

os.makedirs("logs/", exist_ok = True)
with open(log_filename, "w", encoding = "utf-8") as f:
    f.write(json_str)
print(f"Saved log to {log_filename}")
