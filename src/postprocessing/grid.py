#!/usr/bin/env python

import argparse

from PIL import Image, ImageDraw

if __name__ == '__main__':
    N = 8
    image = Image.new(mode='RGBA', size=(641, 641), color=(255, 255, 255))

    # Draw a grid
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height - 1
    cell_size = int(image.width / N)

    for x in range(0, image.width, cell_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=128)

    x_start = 0
    x_end = image.width - 1

    for y in range(0, image.height, cell_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=128)
        print(y)
    del draw

    hand = Image.open('hand.png').convert("RGBA")
    hand = hand.resize((cell_size, cell_size), Image.ANTIALIAS)
    image.paste(hand, box=(0*cell_size, 0*cell_size), mask=hand)

    hand = Image.open('light.png').convert("RGBA")
    hand = hand.resize((cell_size, cell_size), Image.ANTIALIAS)
    image.paste(hand, box=(6 * cell_size, 1 * cell_size), mask=hand)

    hand = Image.open('key.png').convert("RGBA")
    hand = hand.resize((cell_size, cell_size), Image.ANTIALIAS)
    image.paste(hand, box=(1 * cell_size, 6 * cell_size), mask=hand)

    hand = Image.open('chest.png').convert("RGBA")
    hand = hand.resize((cell_size, cell_size), Image.ANTIALIAS)
    image.paste(hand, box=(5 * cell_size, 5 * cell_size), mask=hand)
    # image.show()

    image = image.convert('L')
    filename = "/home/pierre/Latex/Papiers_blogs/ICML2018/grid.jpeg"
    image.save(filename)