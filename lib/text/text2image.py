# coding: utf-8

import os
from PIL import Image, ImageFont, ImageDraw
# import pygame

# pygame.init()
# font = pygame.font.Font(r"/Library/Fonts/Arial Unicode.ttf", 36)
font = ImageFont.truetype(r"/Library/Fonts/Arial Unicode.ttf", 36)


def pygame_text2image(text, save_filename):
    rtext = font.render(text, True, (0, 0, 0), (255, 255, 255))
    pygame.image.save(rtext, save_filename)


def pil_text2image(text, save_filename):
    im = Image.new("RGB", (144, 36), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((0, -5), text, font=font, fill="#000000")
    im.save(save_filename)


def main():
    # 为了方便与cw2word对比,取中文字符
    f = open("../../data/words_stroke.txt")
    for i in f:
        i = i.strip().split("\t")
        text = i[0].decode("utf-8")
        # pygame_text2image(text, os.path.join("../../data/text_image", text+".png"))
        if len(text) <= 4:
            pil_text2image(text, os.path.join("../../data/text_image_pil_length_4", text+".png"))


if __name__ == '__main__':
    # pygame_text2image(u"推荐", "t.png")
    # pil_text2image(u"人工智能", "t.png")
    main()