# This is a sample Python script.
import math


def print_hi(name):
    """Print a greeting."""
    print(f'Hi, {name}')


def circle_area(radius):
    """Calculate the area of a circle given its radius."""
    if radius < 0:
        raise ValueError('Radius cannot be negative')
    return math.pi * radius ** 2


if __name__ == '__main__':
    print_hi('PyCharm')

    # Example usage of circle_area
    r = 5
    area = circle_area(r)
    print(f'The area of a circle with radius {r} is: {area:.2f}')
