from adt import fold3, fold4, Sum2, Sum3, Sum4
from adt import F1, F2
from typing import Callable, Dict, List, Type, TypeVar, Union

A = TypeVar('A')

# from https://gist.github.com/catb0t/bd82f7815b7e95b5dd3c3ad294f3cbbf
JsonPrimitive = Sum4[str, int, bool, None]
JsonType = Sum3[JsonPrimitive, 'JsonList', 'JsonDict']

Parsed = Sum2[Exception, A]
Parser = Callable[[JsonType], Parsed[A]]

# work around mypy#731: no recursive structural types yet
class JsonList(List[JsonType]):
    pass

class JsonDict(Dict[str, JsonType]):
    pass

def error(t: Type[A]) -> Callable[[str], Exception]:
  def x(s: str) -> Exception:
    return TypeError('Expecting ' + t.__name__ + ', got ' + s)
  return x

def prims(t: Type[A]) -> fold4[str, int, bool, None, Parsed[A]]:
  err = error(t)
  return fold4[str, int, bool, None, Parsed[A]](
    (lambda x: F1(err('str')),
    lambda x: F1(err('int')),
    lambda x: F1(err('bool')),
    lambda x: F1(err('None'))))


def parsePrim(t: Type[A], 
           f: fold4[str, int, bool, None, Parsed[A]]) -> Parser[A]:
  err = error(t)
  return fold3[JsonPrimitive, 'JsonList', 'JsonDict', Parsed[A]](
    (f,
     lambda x: F1(err('List')),
     lambda x: F1(err('Dict'))))

def parseStr() -> Parser[str]:
  (_, b, c, d) = prims(str).fold
  prim = fold4[str, int, bool, None, Parsed[str]]((lambda x: F2(x), b, c, d))
  err = error(str)
  return parsePrim(str, prim)

def parseInt() -> Parser[int]:
  (a, _, c, d) = prims(int).fold
  prim = fold4[str, int, bool, None, Parsed[int]]((a, lambda x: F2(x), c, d))
  err = error(str)
  return parsePrim(int, prim)

def parseBool() -> Parser[bool]:
  (a, b, _, d) = prims(bool).fold
  prim = fold4[str, int, bool, None, Parsed[bool]]((a, b, lambda x: F2(x), d))
  err = error(bool)
  return parsePrim(bool, prim)

def traverse(keys: List[str]) -> Parser[JsonType]: pass
  #def x(j: JsonType)
