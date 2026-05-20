from math import sqrt, ceil, radians, sin, cos
import numpy as np
import pygame
from random import uniform


"""
    Configure the generation
"""
WIDTH = 1000  # (in pixels)
HEIGHT = 600
TILE_RADIUS = 7   # radius of hexagon (center to vertex)
Z_STEP = 12 # Height difference per unit shifted upwards (smaller step = bigger height difference)


"""
    Compute globals and constants
"""
HEX_WIDTH = TILE_RADIUS * 2
HEX_HEIGHT = TILE_RADIUS * sqrt(3)

X_SPACER = HEX_WIDTH * 0.75
Y_SPACER = HEX_HEIGHT

X_OFFSET = -X_SPACER // 2
Y_OFFSET = -Y_SPACER // 2

_num_tiles = ceil(max(HEIGHT / HEX_HEIGHT, WIDTH / HEX_WIDTH * sqrt(3)))
N = len(bin(_num_tiles - 1)) - 2  # Number of bits

VALUE_RANGE = [0, 255]  # Initial corner value range 0..255
RANDOM_SCALAR = 2 ** (N + 1)  # Initial random noise value -r..r
RANDOM_DECAY = 0.5  # Random scalar decay factor per iteration of diamond-square generation


"""
    Generate terrain heights
"""
def diamond_square(size: int, random_scalar: int) -> '[size][size]uint8':
    grid = np.zeros((size, size), dtype=np.uint8)
    
    # Seed corners with random value
    grid[0,      0     ] = uniform(*VALUE_RANGE)
    grid[0,      size-1] = uniform(*VALUE_RANGE)
    grid[size-1, 0     ] = uniform(*VALUE_RANGE)
    grid[size-1, size-1] = uniform(*VALUE_RANGE)
    
    step_size = size - 1
    r = random_scalar

    while step_size > 1:
        half_step = step_size // 2

        # Diamond step (fill in center of 4 points of square)
        for x in range(0, size-1, step_size):
            for y in range(0, size-1, step_size):
                avg = sum((
                    grid[x,             y],
                    grid[x + step_size, y],
                    grid[x,             y + step_size],
                    grid[x + step_size, y + step_size]
                    )) / 4
                grid[x + half_step, y + half_step] = np.clip(avg + uniform(-r, r), 0, 255)

        # Square step (fill in center of 2-4 points of diamond)
        for x in range(0, size, half_step):
            y_start = half_step if (x % step_size == 0) else 0
            for y in range(y_start, size, step_size):
                total = count = 0    
                
                for nx, ny in [(x - half_step, y), (x + half_step, y), (x, y - half_step), (x, y + half_step)]:
                    if 0 <= nx < size and 0 <= ny < size:
                        total += grid[nx, ny]
                        count += 1
                
                grid[x, y] = np.clip((total / count) + uniform(-r, r), 0, 255)

        r *= RANDOM_DECAY
        step_size //= 2

    # Clip to 0..255
    grid = np.clip(grid, 0, 255)
    return grid


""" Map each height 0..255 to one of 16 colors """
COLOR_CUTOFF = {
    # Water (blue)
     16 : [ 51,  83, 222],  32 : [ 47, 118, 204],
     48 : [ 75, 147, 235],  64 : [ 90, 158, 242],
    # Land (green)
     80 : [ 16, 173,  44],  96 : [  9, 153,  35],
    112 : [  9, 125,  30], 128 : [  9,  97,  25],
    # Mountain (brown)
    144 : [158, 116,   8], 160 : [145, 108,  12],
    176 : [115,  86,  10], 192 : [ 89,  66,   4],
    # Snow cap (white)
    208 : [175, 175, 175], 224 : [196, 196, 196],
    240 : [217, 217, 217], 256 : [238, 238, 238]
}
def height_to_color(heightmap: '[][]uint8') -> '[][]RGB':
    size = heightmap.shape[0]
    color_map = np.zeros((size, size, 3), dtype=np.uint8)
    
    for x in range(size):
        for y in range(size):
            h = heightmap[x, y]
            for cutoff, color in COLOR_CUTOFF.items():
                if h < cutoff:
                    color_map[x, y] = np.array(color)
                    break
    return color_map

def get_hex_points(cx, cy, radius):
    """Calculates the 6 vertices of a flat-topped hexagon."""
    points = []
    for i in range(6):
        angle_deg = 60 * i
        angle_rad = radians(angle_deg)
        px = cx + radius * cos(angle_rad)
        py = cy + radius * sin(angle_rad)
        points.append((px, py))
    return points

class TerrainGenerator:
    __slots__ = ('size', 'random_scalar', 'dis')

    def __init__(self, size, random_scalar, dis):
        self.size = size
        self.random_scalar = random_scalar
        self.dis = dis
        
        self.generate_new()

    def generate_new(self):
        """Generates a new heightmap and updates the existing plots."""
        heightmap = diamond_square(self.size, self.random_scalar)
        color_map = height_to_color(heightmap)
        
        self.dis.fill((0, 0, 0))
        
        size = heightmap.shape[0]
        for y in range(size):
            for x in range(size):
                h = heightmap[x, y]
                color = color_map[x, y]
                
                _x = x * X_SPACER + X_OFFSET
                _y = (y - h/Z_STEP) * Y_SPACER + Y_OFFSET
                if (x & 1): _y += Y_SPACER // 2
                
                pts = get_hex_points(_x, _y, TILE_RADIUS)
                
                pygame.draw.polygon(self.dis, color, pts)
                pygame.draw.polygon(self.dis, (0, 0, 0), pts, width=1)
        
        pygame.display.flip()

# --- Configuration & Run ---
def main():
    GRID_SIZE = (1 << N) + 1
    
    dis = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("3D Terrain Visualization")
    
    world = TerrainGenerator(GRID_SIZE, RANDOM_SCALAR, dis)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    world.generate_new()
    pygame.quit()

if __name__ == '__main__':
    main()
