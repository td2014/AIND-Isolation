"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
###import sample_players as sp #for testing
import math


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    score = custom_score1(game, player)
    return score


def custom_score1(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Custom Heuristic  
    # PlayerCentralityRatio
    
    #
    #
    #

    myLocation = game.get_player_location(player)
    oppLocation = game.get_player_location(game.get_opponent(player))
    rowCenter = game.height//2
    colCenter = game.width//2
#    print("RowCenter, ColCenter = ", rowCenter, colCenter)
#    print("myLocation[0] = ", myLocation[0])
#    print("myLocation[1] = ", myLocation[1])
#    print("oppLocation[0] = ", oppLocation[0])
#    print("oppLocation[1] = ", oppLocation[1])
    
    myDistToCenter=(myLocation[0]-rowCenter)**2 + (myLocation[1]-colCenter)**2
    myDistToCenter=math.sqrt(myDistToCenter) 

    oppDistToCenter=(oppLocation[0]-rowCenter)**2 + (oppLocation[1]-colCenter)**2
    oppDistToCenter=math.sqrt(oppDistToCenter)        
           
#    print("myDistToCenter = ", myDistToCenter)
#    print("oppDistToCenter = ", oppDistToCenter)
    
    myDistToCenter = max(1.0, myDistToCenter)
    oppDistToCenter = max(1.0, oppDistToCenter)
    
    score = oppDistToCenter/myDistToCenter
#    print("score = ", score)
    
#    print(game.to_string())

    input("press key to continue.")
    

    return score


def custom_score2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Custom Heuristic  
    # InverseAvgOpenRadius
    
    #
    #
    #
    
    # 2)  Average reciprocal distance of open squares from center: OSD:
    # 2a) Get list of open squares using utility function
    # 2b) Compute average distance, with minimum bounded by 1.0
    # 2c) Compute reciprocal and assign to score
                      
    score = 1.0       
    return score


def custom_score3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    # Custom Heuristic  
    # CenterToPeripheralOpenRatio
    
    #
    # Ratio of central open squares to central peripheral squares
    #
          
    score = 1.0
    return score

    

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if len(legal_moves)==0:
            return (-1,-1)
            
        # If board uninitialized, select center position as default.    
        if game.move_count==0:
            start_row = game.height // 2
            start_col = game.width // 2
            return (start_row, start_col)
        
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
 
            if self.method=="minimax":
                # set initialization from the point of view of maximizing player 
                opt_score_result = float("-inf")
                opt_move = (-1,-1)

                # if iterative, initialize depth counter to 1                 
                if self.iterative:
                    depth=1
                else:
                    depth=self.search_depth
            
                while True:
                    score_result, test_move = self.minimax(game, depth)
                    if score_result > opt_score_result:
                        opt_score_result = score_result
                        opt_move = test_move
                    
                    # iterate until we hit timeout or break after fixed depth.
                    if self.iterative:
                        depth=depth+1
                    else:
                        break  #done with minimax to fixed depth
                        
            elif self.method=="alphabeta":
                
                opt_score_result = float("-inf")
                opt_move = (-1,-1)
                
                # if iterative, initialize depth counter to 1 
                if self.iterative:
                    depth=1
                else:
                    depth=self.search_depth
            
                while True:
                    score_result, test_move = self.alphabeta(game, depth)
                    if score_result > opt_score_result:
                        opt_score_result = score_result
                        opt_move = test_move
                    
                    # iterate until we hit timeout or break after fixed depth.
                    if self.iterative:
                        depth=depth+1
                    else:
                        break  #done with alphabeta to fixed depth
            
        except Timeout:
            # Handle any actions required at timeout, if necessary
            return opt_move

        # Return the best move from the last completed search iteration
        return opt_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
       
        # using depth to determine level to evaluate
        # score.  When depth==1 then stop recursing.
        
        # if no legal moves, return with defaults.
        if len(game.get_legal_moves()) == 0:
            if maximizing_player:
                return float("-inf"), (-1,-1)
            else:
                return float("inf"), (-1,-1)
        else:
        # initialize return scores and moves
            if maximizing_player:
                opt_score_result = float("-inf")
                opt_move = (-1,-1)
            else:
                opt_score_result = float("inf")
                opt_move = (-1,-1)  
                
        # we are at target depth
        # now loop over legal moves and determine max or min scoring move
        # depending on layer type.        
            if depth == 1:
                # Loop over legal moves
                for iMove in game.get_legal_moves():
                    gameTemp= game.forecast_move(iMove)
                    score_result = self.score(gameTemp, self)
                    if maximizing_player:
                        if score_result > opt_score_result:
                            opt_score_result = score_result
                            opt_move = iMove
                        else:
                            continue
                    else: # minimizing player process
                        if score_result < opt_score_result:
                            opt_score_result = score_result
                            opt_move = iMove
                        else:
                            continue
                return opt_score_result, opt_move
            else:
                # recurse to the next level down
                # Loop over legal moves
                for iMove in game.get_legal_moves():
                    # update game board with parent move before recursing.
                    gameTemp = game.forecast_move(iMove)
                    # recursive call:  decrease depth and invert maximize to toggle between min/max layers
                    score_result, test_move = self.minimax(gameTemp, depth-1, not maximizing_player)
                    # want to update max or min depending if current layer is maximizing or minimizing
                    if maximizing_player:
                        if score_result > opt_score_result:
                            opt_score_result = score_result
                            opt_move = iMove
                        else:
                            continue
                    else:  # minimizing player processing
                        if score_result < opt_score_result:
                            opt_score_result = score_result
                            opt_move = iMove
                        else:
                            continue       
                return opt_score_result, opt_move
            

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        
#
# Algorithm to implement alpha beta pruning.
#
        
        # using depth to determine level to evaluate
        # score.  When depth==1 then stop recursing.
            
        # if no legal moves immediately return defaults.
        if len(game.get_legal_moves()) == 0:
            if maximizing_player:
                return float("-inf"), (-1,-1)
            else:
                return float("inf"), (-1,-1)
        else:
        # initialize return scores and moves
            if maximizing_player:
                opt_score_result = float("-inf")
                opt_move = (-1,-1)
            else:
                opt_score_result = float("inf")
                opt_move = (-1,-1)  
                
        # we are at target depth
        # now loop over legal moves and determine max or min scoring move
        # depending on layer type.        
            if depth == 1:
                # get legal moves to loop over
                game_legal_moves=game.get_legal_moves()
                for iMove in game_legal_moves:
                    # update board with move and score
                    gameTemp=game.forecast_move(iMove)
                    score_result = self.score(gameTemp, self)
                    if maximizing_player:
                        # check for pruning opportunity
                        if score_result >= beta:
                                opt_score_result=score_result
                                opt_move=iMove
                                break
                        #else continue regular minimax processing
                        if score_result > opt_score_result:
                            opt_score_result = score_result
                            opt_move = iMove
                        else:
                            continue
                    else:  # minimizing player
                        # check for pruning opportunity
                        if score_result <= alpha:
                                opt_score_result=score_result
                                opt_move=iMove
                                break
                        #else continue regular minimax processing
                        if score_result < opt_score_result:
                            opt_score_result = score_result
                            opt_move = iMove
                        else:
                            continue
                return opt_score_result, opt_move
            else:
                # recurse to the next level down
                # Loop over legal moves
                game_legal_moves=game.get_legal_moves()
                for iMove in game_legal_moves:
                    # update game board with parent move before recursing.
                    gameTemp = game.forecast_move(iMove)
                    # recursive call:  decrease depth and invert maximize to toggle between min/max layers
                    score_result, test_move = self.alphabeta(gameTemp, depth-1, alpha, beta, not maximizing_player)
                    # want to update max or min depending if current layer is maximizing or minimizing
                    if maximizing_player:
                        # check for pruning opportunity else update alpha or optimal score/move
                        if score_result >= beta:
                            opt_score_result = score_result
                            opt_move = iMove
                            break
                        elif score_result > alpha:
                            alpha=score_result
                            opt_score_result = score_result
                            opt_move = iMove
                        elif score_result > opt_score_result:
                            opt_score_result = score_result
                            opt_move = iMove
                        else:
                            continue
                    else:
                        # check for pruning opportunity else update beta or optimal score/move
                        if score_result <= alpha:
                            opt_score_result = score_result
                            opt_move = iMove
                            break                     
                        elif score_result < beta:
                            beta=score_result
                            opt_score_result = score_result
                            opt_move = iMove
                        elif score_result < opt_score_result:
                            opt_score_result = score_result
                            opt_move = iMove
                        else:
                            continue   
                        
                return opt_score_result, opt_move
