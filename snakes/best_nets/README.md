# What the AI sees:
    - 8 orthogonal projections from the head for 3 different objectives.
    - Inputs are normalized:
    -- distance / board_size.

    - 8 For walls around the head
    - 8 For body around the head
    - 8 For apple around the head


# Model Architecture:
    - Weights Initialized [-1 to 1]

    - Layers:
        oooo oooo... x 24
        \\\\ ////
         ooo ooo... x 20
         \\\ ///
         ooo ooo... x 18
         \\\ ///
            o... x 3


    1.    :  24 ~~~~ Inputs (8 * 3)
    1.5.  :  SIGMOID ACTIVATION
    
    2.    :  20 ~~~~ Hidden_1
    2.5.  :  SIGMOID ACTIVATION
    
    3.    :  18 ~~~~ Hidden_2
    3.5.  :  SIGMOID ACTIVATION
    
    4.    :  3 ~~~~~ Output
    4.5.  :  SIGMOID ACTIVATION
