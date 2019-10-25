from adt import append2sg, fold3, fold4, map2, Sum2, Sum3, Sum4
from adt import F1, F2, ListSg
from typing import Callable, Dict, Generic, List, Tuple, Type, TypeVar, Union

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
Parser = Callable[[JsonType], Parsed[A]]

# work around mypy#731: no recursive structural types yet
class JsonList(List[JsonType]):
    pass

class JsonDict(Dict[str, JsonType]):
    pass

def error(t: Type[A]) -> Callable[[str], List[Exception]]:
  def x(s: str) -> List[Exception]:
    return [TypeError('Expecting ' + t.__name__ + ', got ' + s)]
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

def parseNone() -> Parser[None]:
  def x(j: JsonType) -> Parsed[None]:
    return F2(None)
  return x

class Box(Generic[A]):

  def __init__(self, a: A) -> None:
    self.a = a

  def run(self) -> A:
    return self.a

class Combine2(Generic[A, B, C]):
  pa: Box[Parser[A]]
  pb: Box[Parser[B]]
  abc: Box[Callable[[Tuple[A, B]], C]]

  def __call__(self) -> Parser[C]:
    def x(t: Tuple[A, B, None, None, None, None, None]) -> C:
      return self.abc.run()((t[0], t[1]))
    return Combine7(self.pa.run(), self.pb.run(), parseNone(), parseNone(), 
                    parseNone(), parseNone(), parseNone(), x)()

class Combine3(Generic[A, B, C, D]):
  pa: Box[Parser[A]]
  pb: Box[Parser[B]]
  pc: Box[Parser[C]]
  abc: Box[Callable[[Tuple[A, B, C]], D]]

  def __call__(self) -> Parser[D]:
    def x(t: Tuple[A, B, C, None, None, None, None]) -> D:
      return self.abc.run()((t[0], t[1], t[2]))
    return Combine7(self.pa.run(), self.pb.run(), self.pc.run(), parseNone(), 
                    parseNone(), parseNone(), parseNone(), x)()

class Combine4(Generic[A, B, C, D, E]):
  pa: Box[Parser[A]]
  pb: Box[Parser[B]]
  pc: Box[Parser[C]]
  pd: Box[Parser[D]]
  abc: Box[Callable[[Tuple[A, B, C, D]], E]]

  def __call__(self) -> Parser[E]:
    def x(t: Tuple[A, B, C, D, None, None, None]) -> E:
      return self.abc.run()((t[0], t[1], t[2], t[3]))
    return Combine7(self.pa.run(), self.pb.run(), self.pc.run(), self.pd.run(),
                    parseNone(), parseNone(), parseNone(), x)()

class Combine5(Generic[A, B, C, D, E, F]):
  pa: Box[Parser[A]]
  pb: Box[Parser[B]]
  pc: Box[Parser[C]]
  pd: Box[Parser[D]]
  pe: Box[Parser[E]]
  abc: Box[Callable[[Tuple[A, B, C, D, E]], F]]

  def __call__(self) -> Parser[F]:
    def x(t: Tuple[A, B, C, D, E, None, None]) -> F:
      return self.abc.run()((t[0], t[1], t[2], t[3], t[4]))
    return Combine7(self.pa.run(), self.pb.run(), self.pc.run(), self.pd.run(),
                    self.pe.run(), parseNone(), parseNone(), x)()

class Combine6(Generic[A, B, C, D, E, F, G]):
  pa: Box[Parser[A]]
  pb: Box[Parser[B]]
  pc: Box[Parser[C]]
  pd: Box[Parser[D]]
  pe: Box[Parser[E]]
  pf: Box[Parser[F]]
  abc: Box[Callable[[Tuple[A, B, C, D, E, F]], G]]

  def __call__(self) -> Parser[G]:
    def x(t: Tuple[A, B, C, D, E, F, None]) -> G:
      return self.abc.run()((t[0], t[1], t[2], t[3], t[4], t[5]))
    return Combine7(self.pa.run(), self.pb.run(), self.pc.run(), self.pd.run(),
                    self.pe.run(), self.pf.run(), parseNone(), x)()

errAcc = ListSg[Exception]()

class Combine7(Generic[A, B, C, D, E, F, G, H]):
  pa: Box[Parser[A]]
  pb: Box[Parser[B]]
  pc: Box[Parser[C]]
  pd: Box[Parser[D]]
  pe: Box[Parser[E]]
  pf: Box[Parser[F]]
  pg: Box[Parser[G]]
  abc: Box[Callable[[Tuple[A, B, C, D, E, F, G]], H]]

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], pe: Parser[E], pf: Parser[F], pg: Parser[G],
               abc: Callable[[Tuple[A, B, C, D, E, F, G]], H]) -> None:
    (self.pa, self.pb, self.pc, self.pd) = (Box(pa), Box(pb), Box(pc), Box(pd))
    (self.pe, self.pf, self.pg, self.abc) = (Box(pe), Box(pf), Box(pg), Box(abc))
    
  def __call__(self) -> Parser[H]:
    def x(j: JsonType) -> Parsed[H]:
      fg = map2(append2sg(self.pf.run()(j), self.pg.run()(j), errAcc), lambda x: x)
      efg = map2(append2sg(self.pe.run()(j), fg, errAcc), lambda x: (x[0],) +  x[1])
      defg = map2(append2sg(self.pd.run()(j), efg, errAcc), lambda x: (x[0],) + x[1])
      cdefg = map2(append2sg(self.pc.run()(j), defg, errAcc), lambda x: (x[0],) + x[1])
      bcdefg = map2(append2sg(self.pb.run()(j), cdefg, errAcc), lambda x: (x[0],) + x[1])
      abcdefg = map2(append2sg(self.pa.run()(j), bcdefg, errAcc), lambda x: (x[0],) + x[1])
      return map2(abcdefg, self.abc.run())
    return x

def traverse(keys: List[str]) -> Parser[JsonType]: pass
  #def x(j: JsonType)
