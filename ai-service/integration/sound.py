import pygame

pygame.mixer.init()
pygame.mixer.music.load("output.mp3")  # or "output.wav"
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
