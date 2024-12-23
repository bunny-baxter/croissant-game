import toml

config = toml.load("game_config.toml")

class GameEndedException(Exception):
    pass

class InvalidActionException(Exception):
    pass

class CroissantGame():
    def __init__(self):
        self.money = 0
        self.investments = []
        self.croissants = 0
        self.turns_left = config["turns"]

    def _end_turn(self):
        self.turns_left -= 1
        for i in range(len(self.investments)):
            self.investments[i] -= 1
            if self.investments[i] == 0:
                self.money += config["invest_payoff"]
        self.investments = list(filter(lambda t: t > 0, self.investments))

    def execute_labor(self):
        if self.turns_left <= 0:
            raise GameEndedException()
        self.money += config["labor_payoff"]
        self._end_turn()

    def execute_invest(self):
        if self.turns_left <= 0:
            raise GameEndedException()
        if self.money < config["invest_cost"]:
            raise InvalidActionException(f'You need {config["invest_cost"]} dollars to invest but you only have {self.money} dollars.')
        self.money -= config["invest_cost"]
        self.investments.append(config["invest_lag_turns"])
        self._end_turn()

    def execute_consume(self, cost):
        if self.turns_left <= 0:
            raise GameEndedException()
        if not cost in config["consume_costs"]:
            raise InvalidActionException(f'You can\'t buy {cost} croissants.')
        if self.money < cost:
            raise InvalidActionException(f'You need {cost} dollars to buy {cost} croissants, but you only have {self.money} dollars.')
        self.money -= cost
        self.croissants += cost
        self._end_turn()
