import math
from typing import Callable, Iterable, List

EPS = 1e-6  # Small constant to avoid division by zero in logarithm and inverse functions


def mul(x: float, y: float) -> float:
    """
    Multiply two numbers.
    f(x, y) = x * y
    Based on the multiplication algorithm.
    """
    return x * y


def id(x: float) -> float:
    """
    Return the identity of the number.
    f(x) = x
    Based on the identity function.
    """
    return x


def add(x: float, y: float) -> float:
    """
    Add two numbers.
    f(x, y) = x + y
    Based on the addition algorithm.
    """
    return x + y


def neg(x: float) -> float:
    """
    Negate the number.
    f(x) = -x
    Based on the negation operation.
    """
    return -x


def lt(x: float, y: float) -> float:
    """
    Check if x is less than y.
    f(x, y) = 1 if x < y else 0
    Based on the less-than comparison.
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """
    Check if x is equal to y.
    f(x, y) = 1 if x = y else 0
    Based on the equality comparison.
    """
    return 1.0 if x == y else 0.0


def max_val(x: float, y: float) -> float:
    """
    Return the maximum of x and y.
    f(x, y) = x if x > y else y
    Based on the max function.
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """
    Check if x is approximately equal to y.
    f(x, y) = |x - y| < 1e-2
    Based on the approximate equality test.
    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """
    Return the sigmoid of x.
    f(x) = 1 / (1 + e^(-x))
    Based on the sigmoid function.
    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """
    Return the rectified linear unit of x.
    f(x) = x if x > 0 else 0
    Based on the ReLU activation function.
    """
    return max(0.0, x)


def log(x: float) -> float:
    """
    Return the natural logarithm of x.
    f(x) = log(x)
    Based on the logarithm function.
    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """
    Return the exponential of x.
    f(x) = e^x
    Based on the exponential function.
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """
    Backward pass for logarithm.
    f'(x) = d / (x + EPS)
    Based on the derivative of the logarithm.
    """
    return d / (x + EPS)


def inv(x: float) -> float:
    """
    Return the inverse of x.
    f(x) = 1 / x
    Based on the inverse function.
    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """
    Backward pass for inverse.
    f'(x) = -d / (x^2)
    Based on the derivative of the inverse function.
    """
    return -d / (x * x) ** 2


def relu_back(x: float, d: float) -> float:
    """
    Backward pass for rectified linear unit.
    f'(x) = d if x > 0 else 0
    Based on the derivative of the ReLU function.
    """
    return d if x > 0 else 0.0


def map_function(fn: Callable[[float], float], ls: Iterable[float]) -> List[float]:
    """
    Apply a function to each element in a list.
    Based on the map function in functional programming.
    """
    return [fn(x) for x in ls]


def neg_list(ls: Iterable[float]) -> List[float]:
    """
    Negate each element in a list.
    Utilizes the map function with negation.
    """
    return map_function(neg, ls)


def zip_with(fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]) -> List[float]:
    """
    Apply a function to pairs of elements from two lists.
    Based on the zipWith function in functional programming.
    """
    return [fn(x, y) for x, y in zip(ls1, ls2)]


def add_lists(ls1: Iterable[float], ls2: Iterable[float]) -> List[float]:
    """
    Add corresponding elements of two lists.
    Utilizes the zipWith function with addition.
    """
    return zip_with(add, ls1, ls2)


def reduce_function(fn: Callable[[float, float], float], start: float, ls: Iterable[float]) -> float:
    """
    Reduce a list to a single value using a binary function.
    Based on the reduce function in functional programming.
    """
    result = start
    for x in ls:
        result = fn(result, x)
    return result


def sum_list(ls: Iterable[float]) -> float:
    """
    Sum up a list.
    Utilizes the reduce function with addition.
    """
    return reduce_function(add, 0.0, ls)


def prod_list(ls: Iterable[float]) -> float:
    """
    Calculate the product of a list.
    Utilizes the reduce function with multiplication.
    """
    return reduce_function(mul, 1.0, ls)
