import cv2
import dlib
import pygame
import numpy as np
import os
import sys
import pygame_gui

# Initialize Pygame
pygame.init()

# Load the custom sound effect
hit_sound = pygame.mixer.Sound('ball.wav')
wall_hit_sound = pygame.mixer.Sound('wall.wav')

# Set up the game window
WIDTH, HEIGHT = 2000, 1200
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Eye-tracking Ping Pong")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (173, 216, 230)

# Paddle settings
PADDLE_WIDTH, PADDLE_HEIGHT = 15, 200
paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2

# Ball settings
BALL_SIZE = 10
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_dx, ball_dy = 5, 5

# Score
score = 0

# Debug mode
DEBUG = True

# Font
font = pygame.font.Font(None, 36)

# Eye tracking setup
detector = dlib.get_frontal_face_detector()

# Look for the shape predictor in the current directory and a few other common locations
shape_predictor_file = "shape_predictor_68_face_landmarks.dat"
possible_locations = [
    shape_predictor_file,
    os.path.join(os.path.expanduser("~"), "Downloads", shape_predictor_file),
    os.path.join("/", "tmp", shape_predictor_file)
]

predictor = None
for location in possible_locations:
    if os.path.exists(location):
        try:
            predictor = dlib.shape_predictor(location)
            print(f"Found shape predictor file at: {location}")
            break
        except RuntimeError:
            print(f"Failed to load shape predictor from: {location}")

if predictor is None:
    print(f"Error: Could not find {shape_predictor_file}. Please ensure it's in the correct location.")
    sys.exit(1)

# Try different video capture devices
for i in range(10):  # Try the first 10 possible devices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Successfully opened video capture device: {i}")
        break
    cap.release()

if not cap.isOpened():
    print("Error: Cannot open webcam. Make sure it's connected and the correct device is specified.")
    sys.exit(1)

def get_eye_position():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = np.mean([(landmarks.part(36).x, landmarks.part(36).y),
                            (landmarks.part(39).x, landmarks.part(39).y)], axis=0)
        right_eye = np.mean([(landmarks.part(42).x, landmarks.part(42).y),
                             (landmarks.part(45).x, landmarks.part(45).y)], axis=0)
        eye_center = np.mean([left_eye, right_eye], axis=0)

        return eye_center

    return None

def update_paddle_position(eye_y, scale_factor):
    global paddle_y
    screen_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    paddle_y = HEIGHT - (eye_y / screen_height) * HEIGHT - int(PADDLE_HEIGHT * scale_factor) // 2
    paddle_y = max(0, min(paddle_y, HEIGHT - int(PADDLE_HEIGHT * scale_factor)))

def draw_rounded_rect(surface, color, rect, radius):
    pygame.draw.rect(surface, color, rect.inflate(-radius * 2, 0))
    pygame.draw.rect(surface, color, rect.inflate(0, -radius * 2))
    pygame.draw.circle(surface, color, rect.topleft, radius)
    pygame.draw.circle(surface, color, rect.topright, radius)
    pygame.draw.circle(surface, color, rect.bottomleft, radius)
    pygame.draw.circle(surface, color, rect.bottomright, radius)

# Initialize the GUI manager
manager = pygame_gui.UIManager((WIDTH, HEIGHT))

# Menu section dimensions
MENU_WIDTH = 500
MENU_HEIGHT = HEIGHT
menu_surface = pygame.Surface((MENU_WIDTH, MENU_HEIGHT))
menu_surface.fill((140, 140, 140))  # Light gray background

# Debug button
debug_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 10), (MENU_WIDTH-20, 80)),
    text='Toggle Debug',
    manager=manager
)

# UI Scale slider
scale_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((10, 100), (MENU_WIDTH-20, 80)),
    start_value=1.0,
    value_range=(0.5, 2.0),
    manager=manager
)

# Speed selection dropdown
speed_dropdown = pygame_gui.elements.UIDropDownMenu(
    options_list=['Slow', 'Normal', 'Fast'],
    starting_option='Normal',
    relative_rect=pygame.Rect((10, 190), (MENU_WIDTH-20, 80)),
    manager=manager
)

running = True
clock = pygame.time.Clock()
scale_factor = 1.0

# Debug surface for eye movement visualization
debug_surface = pygame.Surface((WIDTH // 4, HEIGHT // 4))
debug_surface.fill(BLACK)
debug_surface.set_alpha(200)  # Semi-transparent

while running:
    time_delta = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == debug_button:
                DEBUG = not DEBUG
        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == scale_slider:
                scale_factor = scale_slider.get_current_value()
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == speed_dropdown:
                speed = speed_dropdown.selected_option
                if speed == 'Slow':
                    ball_dx, ball_dy = 3, 3
                elif speed == 'Normal':
                    ball_dx, ball_dy = 5, 5
                elif speed == 'Fast':
                    ball_dx, ball_dy = 8, 8
        manager.process_events(event)

    manager.update(time_delta)

    # Update ball position
    ball_x += ball_dx
    ball_y += ball_dy

    # Ball collision with top and bottom
    if ball_y <= 0 or ball_y >= HEIGHT - BALL_SIZE:
        ball_dy *= -1
        ball_y = max(0, min(ball_y, HEIGHT - BALL_SIZE))  # Prevent clipping
        wall_hit_sound.play()

    # Ball collision with left wall
    if ball_x >= WIDTH - BALL_SIZE:
        ball_dx *= -1
        ball_x = WIDTH - BALL_SIZE  # Prevent clipping
        wall_hit_sound.play()

    # Ball collision with paddle
    if ball_x <= MENU_WIDTH + PADDLE_WIDTH and paddle_y < ball_y < paddle_y + int(PADDLE_HEIGHT * scale_factor):
        ball_dx *= -1
        ball_x = MENU_WIDTH + PADDLE_WIDTH  # Prevent clipping
        score += 1
        hit_sound.play()  # Play the hit sound

    # Ball out of bounds
    if ball_x <= 0:
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        score = max(0, score - 1)

    # Eye tracking and paddle update
    eye_position = get_eye_position()
    if eye_position is not None:
        update_paddle_position(eye_position[1], scale_factor)

        if DEBUG:
            # Update debug surface with eye position
            debug_surface.fill(BLACK)
            scaled_x = int(eye_position[0] * (WIDTH // 4) / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            scaled_y = int(eye_position[1] * (HEIGHT // 4) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            pygame.draw.circle(debug_surface, GREEN, (scaled_x, scaled_y), 5)

    # Draw everything
    screen.fill(LIGHT_BLUE)  # Light blue background

    # Draw field lines
    pygame.draw.line(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
    pygame.draw.circle(screen, WHITE, (WIDTH // 2, HEIGHT // 2), 50, 2)

    # Draw paddle (rounded rectangle)
    draw_rounded_rect(screen, BLUE, pygame.Rect(MENU_WIDTH, paddle_y, PADDLE_WIDTH, int(PADDLE_HEIGHT * scale_factor)), 5)

    # Draw ball (circle)
    pygame.draw.circle(screen, RED, (int(ball_x), int(ball_y)), int(BALL_SIZE * scale_factor) // 2)

    # Display score
    score_text = font.render(f"Score: {score}", True, BLACK)
    score_rect = score_text.get_rect(center=(WIDTH // 2, 30))
    screen.blit(score_text, score_rect)



    # Draw the menu on the left side of the screen
    screen.blit(menu_surface, (0, 0))
    manager.draw_ui(screen)

    # Display debug status and eye movement visualization

    if DEBUG:
        debug_text = font.render("Debug: ON (Press 'D' to toggle)", True, BLACK)
        screen.blit(debug_text, (0, HEIGHT - 40))  # Bottom-left corner
        screen.blit(debug_surface, (0, HEIGHT - HEIGHT // 4 - 50))  # Bottom-left corner for the debug surface
    else:
        debug_text = font.render("Press 'D' for debug", True, BLACK)
        screen.blit(debug_text, (0, HEIGHT - 40))  # Bottom-left corner


    pygame.display.flip()

cap.release()
pygame.quit()
