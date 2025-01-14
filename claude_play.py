import argparse
import datetime
import json

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
        messages = message_log
    )
    # TODO: recursively dict-ify reply so I don't need the default arg to json.dumps below
    message_log.append(reply.__dict__)
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

    # TODO: parse reply and choose correct action
    game.execute_labor()

print(f"Score was {game.croissants} croissants.")

print(message_log)

json_str = json.dumps({
    "system": system_prompt,
    "messages": message_log,
    "final_money": game.money,
    "final_score": game.croissants,
}, indent = 2, default = lambda obj: json.dumps(obj.__dict__))
print(json_str)

date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if args.real_api:
    log_filename = f"logs/claude_nothoughts_nostash_{date_str}.json"
else:
    log_filename = f"logs/fake_{date_str}.json"

# TODO: actually save log
print(f"saved log to {log_filename}")
