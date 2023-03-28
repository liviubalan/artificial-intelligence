
from sample_players import DataPlayer

_WIDTH = 11
_HEIGHT = 9
_SIZE = (_WIDTH + 2) * _HEIGHT - 2

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
        # self.queue.put(random.choice(state.actions()))
        SEARCH_DEPTH_MAX = 4
        EVAL_FUNC = self.centeredness_and_libs
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            bestMove = self.minimax(state=state, depth=SEARCH_DEPTH_MAX, evalFunc=EVAL_FUNC, playerId=self.player_id)
            self.queue.put(bestMove)

    def minimax(self, state, depth, evalFunc, playerId):
        def min_value(state, depth, evalFunc, playerId):
            if state.terminal_test(): return state.utility(playerId)
            if depth <= 0: return evalFunc(state, playerId)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1, evalFunc, playerId))
            return value

        def max_value(state, depth, evalFunc, playerId):
            if state.terminal_test(): return state.utility(playerId)
            if depth <= 0:
                return evalFunc(state, playerId)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1, evalFunc, playerId))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1, evalFunc, playerId))

    def move_difference(self, gameState, playerId):
        player_loc = gameState.locs[playerId]
        opp_loc = gameState.locs[1 - playerId]
        player_libs = gameState.liberties(player_loc)
        opp_libs = gameState.liberties(opp_loc)
        return len(player_libs) - len(opp_libs)

    def centeredness(self, gameState, playerId):
        player_loc = gameState.locs[playerId]
        opp_loc = gameState.locs[1 - playerId]
        player_xy = self.computeXy(player_loc)
        x_centeredness = 5 - abs(5 - player_xy[0])
        y_centeredness = 4 - abs(4 - player_xy[1])
        player_centeredness = x_centeredness + y_centeredness
        opp_xy = self.computeXy(opp_loc)
        opp_x_cent = 5 - abs(5 - opp_xy[0])
        opp_y_cent = 4 - abs(4 - opp_xy[1])
        opp_centeredness = opp_x_cent + opp_y_cent
        return player_centeredness - opp_centeredness

    def centeredness_and_libs(self, gameState, playerId):
        center_score = self.centeredness(gameState, playerId)
        liberties_score = self.move_difference(gameState, playerId)
        return liberties_score * 2 + center_score * 1

    def computeXy(self, ind):
        return (ind % (_WIDTH + 2), ind // (_WIDTH + 2))
