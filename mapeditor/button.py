import pygame


class Button:
    """Simple image button helper for pygame.

    Usage:
        btn = Button(x, y, image_surface, scale=1.0)
        if btn.draw(screen):
            # button clicked
    """

    def __init__(self, x, y, image, scale=1.0, hover_image=None):
        self.x = x
        self.y = y
        self.scale = scale
        self._raw_image = image
        self._raw_hover_image = hover_image
        if scale != 1.0:
            w = int(image.get_width() * scale)
            h = int(image.get_height() * scale)
            self.image = pygame.transform.scale(image, (w, h))
            if hover_image is not None:
                self.hover_image = pygame.transform.scale(hover_image, (w, h))
            else:
                self.hover_image = None
        else:
            self.image = image
            self.hover_image = hover_image
        self.rect = self.image.get_rect(topleft=(x, y))
        self._was_pressed = False

    def draw(self, surface):
        """Draw the button and return True if it was clicked this frame."""
        action = False
        pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(pos)
        if hovered:
            if pygame.mouse.get_pressed()[0] and not self._was_pressed:
                action = True
                self._was_pressed = True
        # Reset _was_pressed when mouse released
        if not pygame.mouse.get_pressed()[0]:
            self._was_pressed = False

        if hovered and self.hover_image is not None:
            surface.blit(self.hover_image, self.rect)
        else:
            surface.blit(self.image, self.rect)
        return action

    def is_hover(self):
        return self.rect.collidepoint(pygame.mouse.get_pos())
