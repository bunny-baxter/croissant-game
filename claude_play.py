import argparse
import datetime
import json
import os

import anthropic

import fake_anthropic_client
import game_model

parser = argparse.ArgumentParser(prog = "Claude plays the croissant game via the Anthropic API.")
parser.add_argument("--real_api", action = "store_true", help = "Use the real API. If not enabled, uses a local fake instead.")
parser.add_argument("--fake_reply_text", help = "If using the local fake API, use this string as the reply.")
parser.add_argument("--fake_error", action = "store_true", help = "If using the local fake API, simulates an error response.")
args = parser.parse_args()

assert(not args.real_api or (not args.fake_reply_text and not args.fake_error))

class ServerError(Exception):
    pass

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
    client = fake_anthropic_client.FakeAnthropicClient(reply_text = reply_text, is_error = args.fake_error)

system_prompt = f"""\
You are playing a game. Responding "Labor" will give you {game_model.config["labor_payoff"]} dollars. Responding "Invest" will \
cost {game_model.config["invest_cost"]} dollars and then give you {game_model.config["invest_payoff"]} dollars \
{game_model.config["invest_lag_turns"]} turns later. You may buy {game_model.config["consume_costs"][0]}, \
{game_model.config["consume_costs"][1]}, or {game_model.config["consume_costs"][2]} croissants for 1 dollar each by responding \
"Consume" and the number of croissants. Your goal is to buy as many croissants as you can.\
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
        max_tokens = 100,
        temperature = 0,
        system = system_prompt,
        messages = preprocess_message_log_for_api()
    )
    message_log.append(value_object_to_dict(reply))
    if reply.type == "message":
        return reply.content[0].text
    else:
        # Probably an error. Stop the game so I can inspect the reply.
        raise ServerError(f"Got error from server: {str(reply)}")

print("SYSTEM PROMPT:")
print(system_prompt)
print()

while game.turns_left > 0:
    print("USER:")
    prompt = f"There are {game.turns_left} turns remaining. You have {game.money} dollars and have {game.croissants} croissants. What is your action? Respond only with the action."
    print(prompt)

    try:
        reply_text = send_user_message(prompt)
    except ServerError as e:
        print(f"\n{e}\n")
        break

    print("MODEL:")
    print(reply_text)
    print()

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
    "system": system_prompt,
    "messages": message_log,
    "final_money": game.money,
    "final_score": game.croissants,
}, indent = 2)

date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if args.real_api:
    log_filename = f"logs/claude_nothoughts_nostash_{date_str}.json"
else:
    log_filename = f"logs/fake_{date_str}.json"

os.makedirs("logs/", exist_ok = True)
with open(log_filename, "w", encoding = "utf-8") as f:
    f.write(json_str)
print(f"Saved log to {log_filename}")
