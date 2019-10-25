from adt import append2sg, bind2, fold2, fold3, fold4, map2, Sum2, Sum3, Sum4
from adt import F1, F2, F3, F4, ListSg
import json
from typing import Callable, Dict, Generic, List, Tuple, Type, TypeVar

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')
E = TypeVar('E')
F = TypeVar('F')
G = TypeVar('G')
H = TypeVar('H')

# from https://gist.github.com/catb0t/bd82f7815b7e95b5dd3c3ad294f3cbbf
JsonPrimitive = Sum4[str, int, bool, None]
JsonType = Sum3[JsonPrimitive, 'JsonList', 'JsonDict']

Parsed = Sum2[List[Exception], A]
ParseFn = Callable[[JsonType], Parsed[A]]

# work around mypy#731: no recursive structural types yet
class JsonList(List[JsonType]):
    pass

class JsonDict(Dict[str, JsonType]):
    pass

def load_json(j: str) -> any: # type: ignore
  return json.loads(j) # type: ignore

def parse_json(x: any) -> Parsed[JsonType]: # type: ignore
  try:
    if isinstance(x, dict): # type: ignore
      return F2(F3(JsonDict(x)))
    elif isinstance(x, list): # type: ignore
      return F2(F2(JsonList(x)))
    else:
      if isinstance(x, str): # type: ignore
        return F2(F1(F1(x)))
      elif isinstance(x, int): # type: ignore
        return F2(F1(F2(x)))
      elif isinstance(x, bool): # type: ignore
        return F2(F1(F3(x)))
      elif isinstance(x, type(None)): # type: ignore
        return F2(F1(F4(x)))
      else:
        return F1([Exception('Invalid JSON')])
  except Exception as e:
    return F1([e])

def error(t: Type[A]) -> Callable[[str], List[Exception]]:
  def x(s: str) -> List[Exception]:
    return [TypeError('Expecting ' + t.__name__ + ', got ' + s)]
  return x

class Parser(Generic[A]):
  
  def __init__(self, path: List[str], run: ParseFn[A]) -> None:
    self.path = path
    self._run = run

  def run(self, j: JsonType) -> Parsed[A]:
    return bind2(traverse(self.path)(j), self._run)

  def setPath(self, path: List[str]) -> 'Parser[A]':
    self.path = path
    return self

  def parse(self, s: str) -> Parsed[A]:
    return bind2(parse_json(load_json(s)), self.run)


def parseDict(path: List[str] = []) -> Parser[JsonDict]:
  err = error(JsonDict)
  return Parser(path, fold3[JsonPrimitive, 'JsonList', 'JsonDict', Parsed[A]](
    (lambda x: F1(err('JsonPrimitive')),
     lambda x: F1(err('List')),
     lambda x: F2(x))))

def prims(t: Type[A]) -> fold4[str, int, bool, None, Parsed[A]]:
  err = error(t)
  return fold4[str, int, bool, None, Parsed[A]](
    (lambda x: F1(err('str')),
    lambda x: F1(err('int')),
    lambda x: F1(err('bool')),
    lambda x: F1(err('None'))))


def parsePrim(t: Type[A], 
           f: fold4[str, int, bool, None, Parsed[A]]) -> ParseFn[A]:
  err = error(t)
  return fold3[JsonPrimitive, 'JsonList', 'JsonDict', Parsed[A]](
    (f,
     lambda x: F1(err('List')),
     lambda x: F1(err('Dict'))))

def parseStr(path: List[str] = []) -> Parser[str]:
  (_, b, c, d) = prims(str).fold
  prim = fold4[str, int, bool, None, Parsed[str]]((lambda x: F2(x), b, c, d))
  err = error(str)
  return Parser(path, parsePrim(str, prim))

def parseInt(path: List[str] = []) -> Parser[int]:
  (a, _, c, d) = prims(int).fold
  prim = fold4[str, int, bool, None, Parsed[int]]((a, lambda x: F2(x), c, d))
  err = error(str)
  return Parser(path, parsePrim(int, prim))

def parseBool(path: List[str] = []) -> Parser[bool]:
  (a, b, _, d) = prims(bool).fold
  prim = fold4[str, int, bool, None, Parsed[bool]]((a, b, lambda x: F2(x), d))
  err = error(bool)
  return Parser(path, parsePrim(bool, prim))

def parseNone(path: List[str] = []) -> Parser[None]:
  (a, b, c, _) = prims(type(None)).fold
  prim = fold4[str, int, bool, None, Parsed[None]]((a, b, c, lambda x: F2(x)))
  err = error(type(None))
  return Parser(path, parsePrim(type(None), prim))

def constNone() -> Parser[None]:
  def x(j: JsonType) -> Parsed[None]:
    return F2(None)
  return Parser([], x)

def parseOptional(p: Parser[A]) -> Parser[Sum2[None, A]]:
  def x(j: JsonType) -> Parsed[Sum2[None, A]]:
    return fold2[List[Exception], A, Parsed[Sum2[None, A]]](
      (lambda _: map2(parseNone(p.path).run(j), lambda x: F1(x)),
       lambda x: F2(F2(x))))(p.run(j))
  return Parser([], x)

class Combine2(Generic[A, B, C]):

  def __init__(self, pa: Parser[A], pb: Parser[B], abc: Callable[[Tuple[A, B]], C]) -> None:
    (self.pa, self.pb, self.abc) = (pa, pb, abc)

  def __call__(self) -> Parser[C]:
    def x(t: Tuple[A, B, None, None, None, None, None]) -> C:
      return self.abc((t[0], t[1]))
    return Combine7(self.pa, self.pb, constNone(), constNone(), 
                    constNone(), constNone(), constNone(), x)()

class Combine3(Generic[A, B, C, D]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               abc: Callable[[Tuple[A, B, C]], D]) -> None:
    (self.pa, self.pb, self.pc, self.abc) = (pa, pb, pc, abc)

  def __call__(self) -> Parser[D]:
    def x(t: Tuple[A, B, C, None, None, None, None]) -> D:
      return self.abc((t[0], t[1], t[2]))
    return Combine7(self.pa, self.pb, self.pc, constNone(), 
                    constNone(), constNone(), constNone(), x)()

class Combine4(Generic[A, B, C, D, E]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], abc: Callable[[Tuple[A, B, C, D]], E]) -> None:
    (self.pa, self.pb, self.pc, self.pd, self.abc) = (pa, pb, pc, pd, abc)

  def __call__(self) -> Parser[E]:
    def x(t: Tuple[A, B, C, D, None, None, None]) -> E:
      return self.abc((t[0], t[1], t[2], t[3]))
    return Combine7(self.pa, self.pb, self.pc, self.pd,
                    constNone(), constNone(), constNone(), x)()

class Combine5(Generic[A, B, C, D, E, F]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], pe: Parser[E],
               abc: Callable[[Tuple[A, B, C, D, E]], F]) -> None:
    (self.pa, self.pb, self.pc, self.pd, self.pe, self.abc) = (pa, pb, pc, pd, pe, abc)

  def __call__(self) -> Parser[F]:
    def x(t: Tuple[A, B, C, D, E, None, None]) -> F:
      return self.abc((t[0], t[1], t[2], t[3], t[4]))
    return Combine7(self.pa, self.pb, self.pc, self.pd,
                    self.pe, constNone(), constNone(), x)()

class Combine6(Generic[A, B, C, D, E, F, G]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], pe: Parser[E], pf: Parser[F],
               abc: Callable[[Tuple[A, B, C, D, E, F]], G]) -> None:
    (self.pa, self.pb, self.pc, self.pd, 
      self.pe, self.pf, self.abc) = (pa, pb, pc, pd, pe, pf, abc)

  def __call__(self) -> Parser[G]:
    def x(t: Tuple[A, B, C, D, E, F, None]) -> G:
      return self.abc((t[0], t[1], t[2], t[3], t[4], t[5]))
    return Combine7(self.pa, self.pb, self.pc, self.pd,
                    self.pe, self.pf, constNone(), x)()

errAcc = ListSg[Exception]()

class Combine7(Generic[A, B, C, D, E, F, G, H]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], pe: Parser[E], pf: Parser[F], pg: Parser[G],
               abc: Callable[[Tuple[A, B, C, D, E, F, G]], H]) -> None:
    (self.pa, self.pb, self.pc, self.pd) = (pa, pb, pc, pd)
    (self.pe, self.pf, self.pg, self.abc) = (pe, pf, pg, abc)
    
  def __call__(self) -> Parser[H]:
    def x(j: JsonType) -> Parsed[H]:
      fg = map2(append2sg(self.pf.run(j), self.pg.run(j), errAcc), lambda x: x)
      efg = map2(append2sg(self.pe.run(j), fg, errAcc), lambda x: (x[0],) +  x[1])
      defg = map2(append2sg(self.pd.run(j), efg, errAcc), lambda x: (x[0],) + x[1])
      cdefg = map2(append2sg(self.pc.run(j), defg, errAcc), lambda x: (x[0],) + x[1])
      bcdefg = map2(append2sg(self.pb.run(j), cdefg, errAcc), lambda x: (x[0],) + x[1])
      abcdefg = map2(append2sg(self.pa.run(j), bcdefg, errAcc), lambda x: (x[0],) + x[1])
      return map2(abcdefg, self.abc)
    return Parser([], x)

def traverse(k: List[str]) -> ParseFn[JsonType]:
  def x(j: JsonType) -> Parsed[JsonType]:
    if len(k) == 0:
      return F2(j)
    else:
      d = parseDict().run(j)
      try:
        aj = map2(d, lambda y: y.get(k[0])) # type: ignore
        pj = bind2(aj, parse_json) # type: ignore
        return bind2(pj, traverse(k[1:]))
      except Exception as e:
        return F1([e])
  return x