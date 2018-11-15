"""
Sample Python/Pygame Programs
Simpson College Computer Science
http://programarcadegames.com/
http://simpson.edu/computer-science/

From:
http://programarcadegames.com/python_examples/f.php?file=maze_runner.py

Explanation video: http://youtu.be/5-SbFanyUkQ

Part of a series:
http://programarcadegames.com/python_examples/f.php?file=move_with_walls_example.py
http://programarcadegames.com/python_examples/f.php?file=maze_runner.py
http://programarcadegames.com/python_examples/f.php?file=platform_jumper.py
http://programarcadegames.com/python_examples/f.php?file=platform_scroller.py
http://programarcadegames.com/python_examples/f.php?file=platform_moving.py
http://programarcadegames.com/python_examples/sprite_sheets/
"""
import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PURPLE = (255, 0, 255)


class Wall(pygame.sprite.Sprite):

    def __init__(self, x, y, width, height, color):
        """ Constructor function """

        # Call the parent's constructor
        super().__init__()

        # Make a BLUE wall, of the size specified in the parameters
        self.image = pygame.Surface([width, height])
        self.image.fill(color)

        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x

class Door(pygame.sprite.Sprite):

    def __init__(self, x, y, width, height, color):
        """ Constructor function """

        # Call the parent's constructor
        super().__init__()

        # Make a BLUE wall, of the size specified in the parameters
        self.image = pygame.Surface([width, height])
        self.image.fill(color)

        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x

        self.open = 1


class Player(pygame.sprite.Sprite):
    """ This class represents the bar at the bottom that the
    player controls """

    # Set speed vector
    change_x = 0
    change_y = 0

    def __init__(self, x, y):
        """ Constructor function """

        # Call the parent's constructor
        super().__init__()

        # Set height, width
        self.image = pygame.Surface([50, 50])
        self.image.fill(WHITE)

        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x

    def changespeed(self, x, y):
        """ Change the speed of the player. Called with a keypress. """
        self.change_x += x
        self.change_y += y

    def move(self, walls, doors, sprites):
        """ Find a new position for the player """

        # Move left/right
        self.rect.x += self.change_x

        # Did this update cause us to hit a wall?
        block_hit_list = pygame.sprite.spritecollide(self, walls, False)
        for block in block_hit_list:
            # If we are moving right, set our right side to the left side of
            # the item we hit
            if self.change_x > 0:
                self.rect.right = block.rect.left
            else:
                # Otherwise if we are moving left, do the opposite.
                self.rect.left = block.rect.right

        door_hit_list = pygame.sprite.spritecollide(self, doors, False)
        for door in door_hit_list:
            if self.change_x > 0:
                if door.open:
                    self.rect.left = door.rect.right
                else:
                    self.rect.right = door.rect.left
            else:
                if door.open:
                    self.rect.right = door.rect.left
                else:
                    self.rect.left = door.rect.right

        # Move up/down
        self.rect.y += self.change_y

        # Check and see if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, walls, False)
        for block in block_hit_list:

            # Reset our position based on the top/bottom of the object.
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            else:
                self.rect.top = block.rect.bottom

        self.change_x = 0
        self.change_y = 0

        sprite_hit_list = pygame.sprite.spritecollide(self, sprites, False)
        for sprite in sprite_hit_list:
            sprite.seen = 1

class Key(pygame.sprite.Sprite):
    """ This class represents the bar at the bottom that the
    player controls """

    def __init__(self, x, y):
        """ Constructor function """

        # Call the parent's constructor
        super().__init__()

        # Set height, width
        self.image = pygame.Surface([50, 50])
        self.image.fill(RED)

        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x

        self.seen = 0

class Chest(pygame.sprite.Sprite):
    """ This class represents the bar at the bottom that the
    player controls """

    def __init__(self, x, y):
        """ Constructor function """

        # Call the parent's constructor
        super().__init__()

        # Set height, width
        self.image = pygame.Surface([50, 50])
        self.image.fill(PURPLE)

        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x

        self.seen = 0

class Room(object):
    """ Base class for all rooms. """

    # Each room has a list of walls, and of enemy sprites.
    wall_list = None
    sprites = None
    door_list = None

    def __init__(self):
        """ Constructor, create our lists. """
        self.wall_list = pygame.sprite.Group()
        self.sprites = pygame.sprite.Group()
        self.door_list = pygame.sprite.Group()


class Room1(Room):
    """This creates all the walls in room 1"""

    def __init__(self):
        super().__init__()
        # Make the walls. (x_pos, y_pos, width, height)

        # This is a list of walls. Each is in the form [x, y, width, height]
        walls = [[0, 0, 25, 600, WHITE],
                 [575, 0, 25, 600, WHITE],
                 [25, 0, 550, 25, WHITE],
                 [25, 575, 550, 25, WHITE],
                 [290, 25, 25, 100, BLUE],
                 [290, 175, 25, 250, BLUE],
                 [290, 475, 25, 100, BLUE]
                 ]

        # Loop through the list. Create the wall, add it to the list
        for item in walls:
            wall = Wall(item[0], item[1], item[2], item[3], item[4])
            self.wall_list.add(wall)

        sprites = [Key(25, 25), Chest(525, 525), Key(25, 525), Chest(525, 25)]
        for item in sprites:
            self.sprites.add(item)

        doors = [[290, 125, 25, 50, GREEN],
                 [290, 425, 25, 50, GREEN]]
        for item in doors:
            door = Door(item[0], item[1], item[2], item[3], item[4])
            self.door_list.add(door)


def main():
    """ Main Program """

    # Call this function so the Pygame library can initialize itself
    pygame.init()

    # Create an 800x600 sized screen
    screen = pygame.display.set_mode([600, 600])

    # Set the title of the window
    pygame.display.set_caption('Maze Runner')

    # Create the player paddle object
    player = Player(25, 25 + 5*50)
    movingsprites = pygame.sprite.Group()
    movingsprites.add(player)

    rooms = []

    room = Room1()
    rooms.append(room)

    current_room_no = 0
    current_room = rooms[current_room_no]

    clock = pygame.time.Clock()

    done = False

    while not done:

        # --- Event Processing ---

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    player.changespeed(-25, 0)
                if event.key == pygame.K_RIGHT:
                    player.changespeed(25, 0)
                if event.key == pygame.K_UP:
                    player.changespeed(0, -25)
                if event.key == pygame.K_DOWN:
                    player.changespeed(0, 25)

            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LEFT:
            #         player.changespeed(-5, 0)
            #     if event.key == pygame.K_RIGHT:
            #         player.changespeed(5, 0)
            #     if event.key == pygame.K_UP:
            #         player.changespeed(0, -5)
            #     if event.key == pygame.K_DOWN:
            #         player.changespeed(0, 5)
            #
            # if event.type == pygame.KEYUP:
            #     if event.key == pygame.K_LEFT:
            #         player.changespeed(5, 0)
            #     if event.key == pygame.K_RIGHT:
            #         player.changespeed(-5, 0)
            #     if event.key == pygame.K_UP:
            #         player.changespeed(0, 5)
            #     if event.key == pygame.K_DOWN:
            #         player.changespeed(0, -5)

        # --- Game Logic ---

        player.move(current_room.wall_list, current_room.door_list)

        # if player.rect.x < -15:
        #     if current_room_no == 0:
        #         current_room_no = 2
        #         current_room = rooms[current_room_no]
        #         player.rect.x = 790
        #     elif current_room_no == 2:
        #         current_room_no = 1
        #         current_room = rooms[current_room_no]
        #         player.rect.x = 790
        #     else:
        #         current_room_no = 0
        #         current_room = rooms[current_room_no]
        #         player.rect.x = 790
        #
        # if player.rect.x > 801:
        #     if current_room_no == 0:
        #         current_room_no = 1
        #         current_room = rooms[current_room_no]
        #         player.rect.x = 0
        #     elif current_room_no == 1:
        #         current_room_no = 2
        #         current_room = rooms[current_room_no]
        #         player.rect.x = 0
        #     else:
        #         current_room_no = 0
        #         current_room = rooms[current_room_no]
        #         player.rect.x = 0

        # --- Drawing ---
        screen.fill(BLACK)

        movingsprites.draw(screen)
        current_room.wall_list.draw(screen)
        current_room.sprites.draw(screen)
        current_room.door_list.draw(screen)

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()