import numpy as np
import random


def augmenter(image):

    choice=random.randint(0,9)

    if choice == 0:
        # flips array left right (vertically)
        return(np.fliplr(image))
    elif choice == 1:
        # flips array up down (horizontically)
        return(np.flipud(image))
    elif choice == 2:
        # moving each element one place clockwise
        return(np.rot90(image, k=1, axes=(1,0)))
    elif choice == 3:
        # moving each element one place counter clockwise
        return(np.rot90(m, k=1, axes=(0,1)))
    elif choice == 4:
        # ...
    elif choice == 5:
        # ...
    elif choice == 6:
        # ...
    else:
        return(image)
    
    


