# PLAY THE GAME OF TIC TAC TOE

#-------Global variables-------
#game board
board=["-","-","-",
       "-","-","-",
       "-","-","-"]

game_going=True
#who wins? or tie
winner = None
#Whose turn is this
current_player="X"

# display the board
def display_board():
  print(board[0] + "|" + board[1] + "|" + board[2])
  print(board[3] + "|" + board[4] + "|" + board[5])
  print(board[6] + "|" + board[7] + "|" + board[8])


def play_game():

    #display initial board
    display_board()

    # To ask position from the user
    def handle_turn(player):
        print(player + "'s turn")
        position = input("Enter the position from 1-9")

        valid=False
        while not valid:
            # check if input is valid
            while position not in ["1","2","3","4","5","6","7","8","9"]:
                position = input("Invalid Input. Enter the position from 1-9")

            position = int(position) - 1
            #check if the space is empty to take the input
            if board[position]=="-":
                valid=True
            else:
                print("you can't go there, Try Again")

        board[position] = player
        display_board()


    while game_going:

        #check if the game has ended
        def check_game_over():
            check_win()
            check_tie()

        def check_win():
            global winner
            #check rows
            row_winner=check_rows()
            #check columns
            column_winner=check_columns()
            #check diagonals
            diagonal_winner=check_diagonals()

            if row_winner:
                winner = row_winner
            elif column_winner:
                winner = column_winner
            elif diagonal_winner:
                winner = diagonal_winner
            else:
                winner = None
            return

        def check_rows():
            global game_going
            #chack if any of the rows has same values
            row1=board[0]==board[1]==board[2] != "-"
            row2=board[3]==board[4]==board[5] != "-"
            row3=board[6] == board[7] == board[8] != "-"

            if row1 or row2 or row3:
                game_going=False
             #Return the Winner X or O
            if row1:
                return board[0]
            elif row2:
                return board[3]
            elif row3:
                return board[6]
            else:
                return None



        def check_columns():
            global game_going
            # chack if any of the columns has same values
            col1 = board[0] == board[3] == board[6] != "-"
            col2 = board[1] == board[4] == board[7] != "-"
            col3 = board[2]== board[5] == board[8] != "-"

            if col1 or col2 or col3:
                game_going = False
            # Return the Winner X or O
            if col1:
                return board[0]
            elif col2:
                return board[1]
            elif col3:
                return board[2]
            else:
                return None
            return

        def check_diagonals():
            global game_going
            # chack if any of the diagonals has same values
            diag1 = board[0] == board[4] == board[8] != "-"
            diag2 = board[2] == board[4] == board[6] != "-"

            if diag1 or diag2:
                game_going = False
            # Return the Winner X or O
            if diag1:
                return board[0]
            elif diag2:
                return board[2]
            else:
                return None




        def check_tie():
            global game_going
            if "-" not in board:
                game_going=False
                return True
            else:
                return False

        #flip to other player
        def flip_player():
            global current_player
            if current_player=='X':
                current_player='O'
            elif current_player=='O':
                current_player='X'
            return


        #ask current user to enter
        handle_turn(current_player)
        #after current user enters check if game is over
        check_game_over()
        #flip the players
        flip_player()

     # The game has ended
    if winner == "X" or winner == "O":
         print(winner + " won")
    elif winner == None:
         print("Tie")


play_game()
