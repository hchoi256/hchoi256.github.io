---
layout: single
title: "ML: Minimax 알고리즘 - Teeko Game"
categories: ML
tag: [machine learning, teeko, game, minimax, python]
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JJUNS"
#author_profile: false
header:
    teaser: /assets/images/posts/teeko.png
sidebar:
    nav: "docs"
---

**Minimax** 알고리즘을 활용해서 **Teeko** 게임을 구현한다.

Teeko 게임은 한국인들에게는 익숙하지 않은 보드 게임일 수도 있다 (하기 설명 참조).

> ***Teeko Game***이란?
>
> It is a game between two players on a 5x5 board. Each player has four markers of either red or black. Beginning with black, they take turns placing markers (the "drop phase") until all markers are on the board, with the goal of getting **four in a row horizontally, vertically, or diagonally, or in a 2x2 box as shown above**. If after the drop phase neither player has won, they continue taking turns moving one marker at a time -- to an adjacent space only! (this includes diagonals, not just left, right, up, and down one space.) -- until one player wins. Note, the game has no “wrap-around” similar to other board games, so a player can not move off of the board or win using pieces on the other side of the board.

상기 설명에 나온 것처럼 4개의 마커가 수직, 수평, 대각, 혹은 2x2 박스 형태로 존재하게 되면 승리한다.

이 게임에서 당신은 AI 로봇과 Teeko 보드 게임을 펼치게 될 것이다.

이 프로젝트는 Minimax 알고리즘을 이해하고 있다는 전제로 진행한다.

> [Minimax 알고리즘](https://github.com/hchoi256/ai-terms/blob/main/README.md)이란?

# Code
**[Notice]** [download here](https://github.com/hchoi256/cs540-AI/tree/main/teeko-minimax-game)
{: .notice--danger}

# TeekoPlayer 클래스

Teeko 게임을 진행함에 있어서 각 플레이어 Object에 대해 제어하기 위해 클래스를 만들어보자.

자세한 설명은 코드 주석으로써 적어놨으니 참고하자.

```python
class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]    

    def GetPiecePos(self, state):
        b = r = []
        for row in range(5):
            for col in range(5):
                if state[row][col] == 'b':
                    b.append((row,col))
                elif state[row][col] == 'r':
                    r.append((row,col))
        return b,r

    # check largest # pieces nearby
    def heuristic_gv(self, state, piece):
        b,r = self.GetPiecePos(state)
        if piece == 'b':
            mine = 'b'
            oppo = 'r'
        elif piece == 'r':
            mine = 'r'
            oppo = 'b'

        # horizontal
        mymax = oppmax = mycnt = oppcnt = 0        
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j] == mine:
                    mycnt += 1
            if mycnt > mymax:
                mymax = mycnt
            mycnt = 0
        
        i = j = 0
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j] == oppo:
                    oppcnt += 1
            if oppcnt > oppmax:
                oppmax = oppcnt
            oppcnt = 0

        # vertical
        for i in range(len(state)):
            for j in range(len(state)):
                if state[j][i] == mine:
                    mycnt += 1
            if mycnt > mymax:
                mymax = mycnt
            mycnt = 0
        
        i = j = 0
        for i in range(len(state)):
            for j in range(len(state)):
                if state[j][i] == oppo:
                    oppcnt += 1
            if oppcnt > oppmax:
                oppmax = oppcnt
            oppcnt = 0

        # for / diagonal
        mycnt = oppcnt = 0
        for row in range(3, 5):
            for i in range(2):
                if state[row][i] == mine or state[row - 1][i + 1] == mine or state[row - 2][i + 2] == mine or state[row - 3][i + 3] == mine:
                    mycnt += 1
                if mycnt > mymax:
                    mymax = mycnt
                mycnt = 0

        row = i = 0
        for row in range(3, 5):
            for i in range(2):
                if state[row][i] == oppo or state[row - 1][i + 1] == oppo or state[row - 2][i + 2] == oppo or state[row - 3][i + 3] == oppo:
                    oppcnt += 1
                if oppcnt > oppmax:
                    oppmax = oppcnt
                oppcnt = 0

        # diagonal
        mycnt = oppcnt = row = i = 0
        for row in range(2):
            for i in range(2):
                if state[row][i] == mine or state[row + 1][i + 1] == mine or state[row + 2][i + 2] == mine or state[row + 3][i + 3] == mine:
                    mycnt += 1
                if mycnt > mymax:
                    mymax = mycnt
                mycnt = 0

        row = i = 0
        for row in range(2):
            for i in range(2):
                if state[row][i] == oppo or state[row + 1][i + 1] == oppo or state[row + 2][i + 2] == oppo or state[row + 3][i + 3] == oppo:
                    oppcnt += 1
                if oppcnt > oppmax:
                    oppmax = oppcnt
                oppcnt = 0

        # 2X2
        mycnt = oppcnt = row = i = 0
        for row in range(4):
            for i in range(4):
                if state[row][i] == mine or state[row][i + 1] == mine or state[row + 1][i] == mine or state[row + 1][i + 1]== mine:
                    mycnt += 1
                if mycnt > mymax:
                    mymax = mycnt
                mycnt = 0

        row = i = 0
        for row in range(4):
            for i in range(4):
                if state[row][i] == oppo or state[row][i + 1] == oppo or state[row + 1][i] == oppo or state[row + 1][i + 1]== oppo:
                    oppcnt += 1
                if oppcnt > oppmax:
                    oppmax = oppcnt
                oppcnt = 0

        if mymax == oppmax:
            return 0, state
        elif mymax > oppmax:
            return mymax/6, state # if mine is longer than opponent, return positive float

        return (-1) * oppmax/6, state # if opponent is longer than mine, return negative float

    def Max_Value(self, state, depth):
        bstate = state
        if self.game_value(state) != 0:
            return self.game_value(state),state
        if depth >= 3:
            return self.heuristic_gv(state,self.my_piece)
                
        a = float('-Inf')
        for s in self.succ(state, self.my_piece):
            val = self.Min_Value(s, depth+1)
            if val[0] > a:
                a = val[0]
                bstate = s        
        
        return a, bstate

    def Min_Value(self, state,depth):
        bstate = state
        if self.game_value(state) != 0:
            return self.game_value(state),state        
        if depth >= 3:
            return self.heuristic_gv(state, self.opp)
    
        b = float('Inf')
        for s in self.succ(state, self.opp):
            val = self.Max_Value(s, depth+1)
            if val[0] < b:
                b = val[0]
                bstate = s

        return b, bstate
    
    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        drop_phase = True   # TODO: detect drop phase        
        if sum((i.count('b') for i in state)) >= 4 and sum((i.count('r') for i in state)) >= 4:
            drop_phase = False                    
        if not drop_phase:
            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!
            move = []
            value, bstate = self.Max_Value(state, 0)            
            arr1 = np.array(state) == np.array(bstate)
            arr2 = np.where(arr1 == False) # check difference between succ and curr state
            if state[arr2[0][0]][arr2[1][0]] == ' ': # find original to define move
                (origrow, origcol) = (arr2[0][1].item(),arr2[1][1].item())
                (row,col) = (arr2[0][0].item(), arr2[1][0].item())
            else:
                (origrow, origcol) = (arr2[0][0].item(), arr2[1][0].item())
                (row, col) = (arr2[0][1].item(), arr2[1][1].item())
            move.insert(0, (row, col))
            move.insert(1, (origrow, origcol))  # move for after drop phase
            return move

        # select an unoccupied space randomly
        # TODO: implement a minimax algorithm to play better       
        move = []
        value, bstate = self.Max_Value(state, 0)
        arr1 = np.array(state) == np.array(bstate)
        arr2 = np.where(arr1 == False) # diff between succ and curr state        
        (row,col) = (arr2[0][0].item(), arr2[1][0].item())
        while not state[row][col] == ' ': # find original to define move
            (row, col) = (arr2[0][0].item(), arr2[1][0].item())

        # ensure the destination (row,col) tuple is at the beginning of the move list
        move.insert(0, (row, col))              
        return move

    def succ(self, state, piece):
        self.game_value(state)        
        succ = []
        drop_phase = True  # TODO: detect drop phase

        if sum((i.count('b') for i in state)) >= 4 and sum((i.count('r') for i in state)) >= 4:
            drop_phase = False
        if not drop_phase:
            for row in range(len(state)):
                for col in range(len(state)):
                    if state[row][col] == piece:
                        succ.insert(0, self.up(state, row, col))
                        succ.insert(1, self.down(state, row, col))
                        succ.insert(2, self.left(state, row, col))
                        succ.insert(3, self.right(state, row, col))
                        succ.insert(4, self.upleft(state, row, col))
                        succ.insert(5, self.upright(state, row, col))
                        succ.insert(6, self.downleft(state, row, col))
                        succ.insert(7, self.downright(state, row, col))
            return list(filter(None, succ))
        for row in range(len(state)):
            for col in range(len(state)):
                new = copy.deepcopy(state)
                if new[row][col] == ' ':
                    new[row][col] = piece
                    succ.append(new)
        return list(filter(None, succ))

    def up(self, k, i, j):
        state = copy.deepcopy(k)
        if i - 1 >= 0 and state[i - 1][j] == ' ':
            state[i][j], state[i - 1][j] = state[i - 1][j], state[i][j]
            return state

    def down(self, k, i, j):
        state = copy.deepcopy(k)
        if i + 1 < len(state) and state[i + 1][j] == ' ':
            state[i][j], state[i + 1][j] = state[i + 1][j], state[i][j]
            return state

    def left(self, k, i, j):
        state = copy.deepcopy(k)
        if j - 1 >= 0 and state[i][j - 1] == ' ':
            state[i][j], state[i][j - 1] = state[i][j - 1], state[i][j]
            return state

    def right(self, k, i, j):
        state = copy.deepcopy(k)
        if j + 1 < len(state) and state[i][j + 1] == ' ':
            state[i][j], state[i][j + 1] = state[i][j + 1], state[i][j]
            return state

    def upleft(self, k, i, j):
        state = copy.deepcopy(k)
        if i - 1 >= 0 and j - 1 >= 0 and state[i - 1][j - 1] == ' ':
            state[i][j], state[i - 1][j - 1] = state[i - 1][j - 1], state[i][j]
            return state

    def upright(self, k, i, j):
        state = copy.deepcopy(k)
        if i - 1 >= 0 and j + 1 < len(state) and state[i - 1][j + 1] == ' ':
            state[i][j], state[i - 1][j + 1] = state[i - 1][j + 1], state[i][j]
            return state

    def downleft(self, k, i, j):
        state = copy.deepcopy(k)
        if i + 1 < len(state) and j - 1 >= 0 and state[i + 1][j - 1] == ' ':
            state[i][j], state[i + 1][j - 1] = state[i + 1][j - 1], state[i][j]
            return state

    def downright(self, k, i, j):
        state = copy.deepcopy(k)
        if i + 1 < len(state) and j + 1 < len(state) and state[i + 1][j + 1] == ' ':
            state[i][j], state[i + 1][j + 1] = state[i + 1][j + 1], state[i][j]
            return state
        
    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins        
        for row in state:            
            for i in range(2):                
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for row in range(2):
            for i in range(2):
                if state[row][i] != ' ' and state[row][i] == state[row + 1][i + 1] == state[row + 2][i + 2] == state[row + 3][i + 3]:                    
                    return 1 if state[row][i] == self.my_piece else -1

        # TODO: check / diagonal wins
        for row in range(3, 5):
            for i in range(2):
                if state[row][i] != ' ' and state[row][i] == state[row - 1][i + 1] == state[row - 2][i + 2] == state[row - 3][i + 3]:                    
                    return 1 if state[row][i] == self.my_piece else -1

        # TODO: check 2x2 box wins
        for row in range(4):
            for i in range(4):
                if state[row][i] != ' ' and state[row][i] == state[row][i + 1] == state[row + 1][i] == state[row + 1][i + 1]:                    
                    return 1 if state[row][i] == self.my_piece else -1    
        return 0 # no winner yet

```

클래스 함수 설명:
- *__init__*: 랜덤으로 Red or Black 마커 결정
- *GetPiecePos*: 5x5 보드판에서 각 마커의 위치를 리스트로 가져온다.
- *heuristic_gv*: 게임 성공 조건 만족시 종료, otherwise Minimax 트리에서 현 노드 값과 현 state 리턴
- *Max_Value*: Minimax 알고리즘에서 Max 값을 도출한다.
- *Min_Value*: Minimax 알고리즘에서 Min 값을 도출한다.
- *make_move*: 자신의 마커가 다음으로 움직일 지점의 row, col 값을 가져온다.
- *succ*: 다음 움직일 경우들의 집합
    - 궁극적으로, 최선의 다음 움직임 선택을 하기 위함.
- *up, down, ..., downright*: 해당 방향으로 마커 움직임
- *opponent_move*: 상대방(AI)의 마커가 다음으로 움직일 지점의 row col 값을 가져온다.
- *place_piece*: 해당 지점으로 주어진 마커를 실제로 이동시킨다.
- *print_board*: 보드판 현황 출력
- *game_value*: 게임 성공 조건 만족 여부 판단

# Teeko 게임 실행 함수

```python
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]: # ai 턴일 경우
            ai.print_board() # 현 보드판 현황
            move = ai.make_move(ai.board) # ai 현 보드 위치에서 다음 위치 얻기
            ai.place_piece(move, ai.my_piece) # 다음 위치로 이동하기
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made: # 플레이어 턴
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))]) # ai의 opponent는 플레이어이므로, 플레이어 마커를 움직인다.
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")
```

상기 코드 역시 주석으로 설명을 추가했지만 주요 부분만 뜯어내서 살펴보자.

'while piece_count < 8 and ai.game_value(ai.board) == 0:'
- 보드판에 놓인 마커 개수가 8개 이상이거나 누군가 성공 조건을 달성했다면 종료한다.

'while ai.game_value(ai.board) == 0:'
- 만약, 보드판에 8개 이상의 마커가 놓였는데도 승자 조건이 만족된 측이 없을 경우 실행되는 부분이다.

# 게임 실행

하기 코드를 통해 간단히 게임 실행이 가능하다.

```python
main()
```    

![image](https://user-images.githubusercontent.com/39285147/184557144-b8d13395-4364-43d4-a312-6a4982288b0c.png)

게임 실행 시 콘솔 환경에서 입력값을 입력하며 게임 진행이 가능하다.

입력값으로 ABCDE 열 중에서 하나를 택하고 row index를 숫자로 조합해서 적어주면 된다 (i.e., 'B3' B열의 4번 째 row에 마커를 놓는다).

하기 사진처럼 B3을 입력하면 보드판이 갱신된다 (***RED: AI | BLACK: 플레이어***)

![image](https://user-images.githubusercontent.com/39285147/184557947-90aad7de-ffb6-413f-8e40-bf2fdca30b35.png)

몇 초가 지나면 AI가 자동적으로 최적의 선택을 하면서 마커를 놓는다 (하기 사진 참조).

![image](https://user-images.githubusercontent.com/39285147/184558029-c985a2b0-cd15-40f0-abaf-4a67492290aa.png)

내가 이상한 곳에 마커를 놓아서, AI가 가장 빠르게 성공 조건을 완성시킬 수 있는 대각선 방향으로 RED 마커를 놓고있다.

각자 게임 진행을 해보면서 코드 또한 다시 한 번 숙지해보길 바란다.