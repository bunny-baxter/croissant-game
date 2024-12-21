import game_model

print("Welcome to the Croissant Game!")

game = game_model.CroissantGame()

def try_action(func):
    try:
        func()
        return True
    except game_model.InvalidActionException as e:
        print(e)
        return False

while game.turns_left > 0:
    print(f"There are {game.turns_left} turns remaining.")
    print(f"You have {game.money} dollars and have eaten {game.croissants} croissants.")
    print(f"1. Labor")
    print(f"2. Invest")
    for i in range(len(game_model.config["consume_costs"])):
        print(f'{3 + i}. Consume-{game_model.config["consume_costs"][i]}')
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
