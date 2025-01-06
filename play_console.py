import argparse

import game_model

parser = argparse.ArgumentParser(prog = "Play the croissant game in the console.")
parser.add_argument("--enable_stash", action = "store_true", help = "Enable the stash, where money can carry over between games.")
args = parser.parse_args()

print("Welcome to the Croissant Game!")

def try_action(func):
    try:
        func()
        return True
    except game_model.InvalidActionException as e:
        print(e)
        return False

stash_value = None
if args.enable_stash:
    stash_value = 0

def play_game_once():
    global stash_value
    game = game_model.CroissantGame(stash_value = stash_value)

    while game.turns_left > 0:
        print(f"There are {game.turns_left} turns remaining.")
        print(f"You have {game.money} dollars and have eaten {game.croissants} croissants.")
        print("1. Labor")
        print("2. Invest")
        for i in range(len(game_model.config["consume_costs"])):
            print(f'{3 + i}. Consume-{game_model.config["consume_costs"][i]}')
        if args.enable_stash:
            print("6. Stash")
            print(f"7. Unstash ({game.stash_value} dollars available)")
        print("> ", end = "")

        pre_action_investments = len(game.investments)

        action = input()
        action = action.strip().lower()
        print()

        found_action = False
        if action == "1" or action == "labor":
            found_action = True
            success = try_action(lambda: game.execute_labor())
            if success:
                print(f'You work hard and get {game_model.config["labor_payoff"]} dollars.')
        elif action == "2" or action == "invest":
            found_action = True
            success = try_action(lambda: game.execute_invest())
            if success:
                print(f'You invest {game_model.config["invest_cost"]} dollars.')
                pre_action_investments += 1 # add one to compensate for the new investment
        elif action == "6" or action == "stash":
            if args.enable_stash:
                found_action = True
                success = try_action(lambda: game.execute_stash())
                if success:
                    print(f'You stash {game_model.config["stash_transfer_value"]} dollars outside the game.')
        elif action == "7" or action == "unstash":
            if args.enable_stash:
                found_action = True
                transfer_value = game.execute_unstash()
                if transfer_value > 0:
                    print(f'You retrieve {transfer_value} dollars from outside the game.')
                else:
                    print("There was no money in the stash.")
        else:
            for i in range(len(game_model.config["consume_costs"])):
                cost = game_model.config["consume_costs"][i]
                if action == str(3 + i) or action == f"consume-{cost}" or action == f"consume{cost}":
                    found_action = True
                    success = try_action(lambda: game.execute_consume(cost))
                    if success:
                        print(f'Yum, fresh croissants!')
                    break
        if not found_action:
            print("I didn't understand that input.")

        if len(game.investments) < pre_action_investments:
            print(f'Your investment paid off. You receive {game_model.config["invest_payoff"]} dollars.')

        print()

    print(f"Game over. You ate {game.croissants} croissants.")

    stash_value = game.stash_value

while True:
    play_game_once()
    print()
    print("Play again? (y/n)")
    while True:
        print("> ", end = "")
        decision = input()
        if decision.lower().startswith("y"):
            print()
            break
        elif decision.lower().startswith("n"):
            exit(0)
        else:
            print("Answer (y)es or (n)o please.")
