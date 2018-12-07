#!/usr/bin/env python

import argparse

from PIL import Image, ImageDraw

if __name__ == '__main__':
    N = 13
    image = Image.new(mode='RGBA', size=(13*40, 13*40), color=(255, 255, 255))

    # Draw a grid
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    cell_size = int(image.width / N)

    for x in range(0, image.width, cell_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=128)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, cell_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=128)

    draw.rectangle(xy=[6 * cell_size, 0 * cell_size, 7 * cell_size, 3 * cell_size], fill='black')
    draw.rectangle(xy=[6 * cell_size, 4 * cell_size, 7 * cell_size, 9 * cell_size], fill='black')
    draw.rectangle(xy=[6 * cell_size, 10 * cell_size, 7 * cell_size, 13 * cell_size], fill='black')
    draw.rectangle(xy=[7 * cell_size, 6 * cell_size, 13 * cell_size, 7 * cell_size], fill='black')
    del draw

    hand = Image.open('hand.png').convert("RGBA")
    hand = hand.resize((cell_size, cell_size), Image.ANTIALIAS)
    image.paste(hand, box=(0 * cell_size, 6 * cell_size), mask=hand)

    key1 = Image.open('key.png').convert("RGBA")
    key1 = key1.resize((cell_size, cell_size), Image.ANTIALIAS)
    image.paste(key1, box=(0 * cell_size, 0 * cell_size), mask=key1)

    key2 = Image.open('key.png').convert("RGBA")
    key2 = key2.resize((cell_size, cell_size), Image.ANTIALIAS)
    image.paste(key2, box=(0 * cell_size, 12 * cell_size), mask=key2)

    chest1 = Image.open('chest.png').convert("RGBA")
    chest1 = chest1.resize((cell_size, cell_size), Image.ANTIALIAS)
    image.paste(chest1, box=(12 * cell_size, 0 * cell_size), mask=chest1)

    chest2 = Image.open('chest.png').convert("RGBA")
    chest2 = chest2.resize((cell_size, cell_size), Image.ANTIALIAS)
    image.paste(chest2, box=(12 * cell_size, 12 * cell_size), mask=chest2)

    door1 = Image.open('door.png').convert("RGBA")
    door1 = door1.resize((cell_size, cell_size), Image.ANTIALIAS)
    image.paste(door1, box=(6 * cell_size, 3 * cell_size), mask=door1)

    door2 = Image.open('door.png').convert("RGBA")
    door2 = door2.resize((cell_size, cell_size), Image.ANTIALIAS)
    image.paste(door2, box=(6 * cell_size, 9 * cell_size), mask=door2)

    image.show()

    image = image.convert('L')
    filename = "/home/pierre/Latex/Papiers_blogs/ICML2018/grid.jpeg"
    image.save(filename)