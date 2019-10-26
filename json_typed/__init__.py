from adt import append2sg, bind2, fold2, fold3, fold4, map2, Sum2, Sum3, Sum4
from adt import Compose, F1, F2, F3, F4, Fn, ListSg
import json
from typing import Callable, Dict, Generic, List, NamedTuple, Tuple, Type, TypeVar

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

ParseError = NamedTuple('ParseError', [('path', List[str]), ('error', Exception)])
Parsed = Sum2[List[ParseError], A]
PreParseError = Sum2[List[ParseError], Exception]
PreParsed = Sum2[PreParseError, A]
ParseFn = Callable[[JsonType], PreParsed[A]]

# work around mypy#731: no recursive structural types yet
class JsonList(List[JsonType]):
    pass

class JsonDict(Dict[str, JsonType]):
    pass

def load_json(j: str) -> any: # type: ignore
  return json.loads(j) # type: ignore

def parse_json(x: any) -> PreParsed[JsonType]: # type: ignore
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
        return F1(F2(Exception('Invalid JSON')))
  except Exception as e:
    return F1(F2(e))

def error(t: Type[A]) -> Callable[[str], PreParseError]:
  return Fn[str, PreParseError](
    lambda s: F2(TypeError('Expecting ' + t.__name__ + ', got ' + s)))

class toParsed(Generic[A]):
  def __init__(self, path: List[str]) -> None:
    self.path = path
    
  def __call__(self, a: PreParsed[A]) -> Parsed[A]:
    return fold2[PreParseError, A, Parsed[A]](
      (fold2[List[ParseError], Exception, Parsed[A]](
        (lambda x: F1(list(map(Fn[ParseError, ParseError](
          lambda y: y._replace(path=self.path+y.path)), x))),
         lambda x: F1([ParseError(self.path, x)]))),
       lambda x: F2(x)))(a)

class toPreParsed(Generic[A]):
  def __call__(self, a: Parsed[A]) -> PreParsed[A]:
    return fold2[List[ParseError], A, PreParsed[A]](
      (lambda x: F1(F1(x)), lambda x: F2(x)))(a)

class Parser(Generic[A]):
  
  def __init__(self, path: List[str], run: ParseFn[A]) -> None:
    self.path = path
    self._run = run

  def run(self, j: JsonType) -> Parsed[A]:
    return bind2(traverse(self.path)(j), Compose(toParsed[A](self.path), self._run))

  def setPath(self, path: List[str]) -> 'Parser[A]':
    self.path = path
    return self

  def parse(self, s: str) -> Parsed[A]:
    return bind2(Compose(toParsed[JsonType](self.path),
                 Compose(parse_json, load_json))(s), self.run)


def parseDict() -> ParseFn[JsonDict]:
  err = error(JsonDict)
  return fold3[JsonPrimitive, 'JsonList', 'JsonDict', PreParsed[A]](
    (lambda x: F1(err('JsonPrimitive')),
     lambda x: F1(err('List')),
     lambda x: F2(x)))

def prims(t: Type[A]) -> fold4[str, int, bool, None, PreParsed[A]]:
  err = error(t)
  return fold4[str, int, bool, None, PreParsed[A]](
    (lambda x: F1(err('str')),
    lambda x: F1(err('int')),
    lambda x: F1(err('bool')),
    lambda x: F1(err('None'))))


def parsePrim(t: Type[A], 
           f: fold4[str, int, bool, None, PreParsed[A]]) -> ParseFn[A]:
  err = error(t)
  return fold3[JsonPrimitive, 'JsonList', 'JsonDict', PreParsed[B]](
    (f,
     lambda x: F1(err('List')),
     lambda x: F1(err('Dict'))))

def parseStr() -> ParseFn[str]:
  (_, b, c, d) = prims(str).fold
  prim = fold4[str, int, bool, None, PreParsed[str]]((lambda x: F2(x), b, c, d))
  return parsePrim(str, prim)

def parseInt() -> ParseFn[int]:
  (a, _, c, d) = prims(int).fold
  prim = fold4[str, int, bool, None, PreParsed[int]]((a, lambda x: F2(x), c, d))
  return parsePrim(int, prim)

def parseBool() -> ParseFn[bool]:
  (a, b, _, d) = prims(bool).fold
  prim = fold4[str, int, bool, None, PreParsed[bool]]((a, b, lambda x: F2(x), d))
  return parsePrim(bool, prim)

def parseNone() -> ParseFn[None]:
  (a, b, c, _) = prims(type(None)).fold
  prim = fold4[str, int, bool, None, PreParsed[None]]((a, b, c, lambda x: F2(x)))
  return parsePrim(type(None), prim)

def parseConst(a: A) -> ParseFn[A]:
  def x(j: JsonType) -> PreParsed[A]:
    return F2(a)
  return x

def const(a: A) -> Parser[A]:
  return Parser([], parseConst(a))

def parseOptional(p: ParseFn[A]) -> ParseFn[Sum2[None, A]]:
  return Fn[JsonType, PreParsed[Sum2[None, A]]](lambda j:
    fold2[PreParseError, A, PreParsed[Sum2[None, A]]](
      (lambda x: fold2[PreParseError, None, PreParsed[Sum2[None, A]]](
          (lambda _: F1(x),
           lambda x: F2(F1(x))))(parseNone()(j)),
       lambda x: F2(F2(x))))(p(j)))

class Combine2(Generic[A, B, C]):

  def __init__(self, pa: Parser[A], pb: Parser[B], abc: Callable[[Tuple[A, B]], C]) -> None:
    (self.pa, self.pb, self.abc) = (pa, pb, abc)

  def __call__(self) -> Parser[C]:
    return Combine7(self.pa, self.pb, const(None), const(None), 
                    const(None), const(None), const(None),
                    Fn[Tuple[A, B, None, None, None, None, None], C](
                      lambda t: self.abc((t[0], t[1]))))()

class Combine3(Generic[A, B, C, D]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               abc: Callable[[Tuple[A, B, C]], D]) -> None:
    (self.pa, self.pb, self.pc, self.abc) = (pa, pb, pc, abc)

  def __call__(self) -> Parser[D]:
    return Combine7(self.pa, self.pb, self.pc, const(None), 
                    const(None), const(None), const(None),
                    Fn[Tuple[A, B, C, None, None, None, None], D](
                      lambda t: self.abc((t[0], t[1], t[2]))))()

class Combine4(Generic[A, B, C, D, E]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], abc: Callable[[Tuple[A, B, C, D]], E]) -> None:
    (self.pa, self.pb, self.pc, self.pd, self.abc) = (pa, pb, pc, pd, abc)

  def __call__(self) -> Parser[E]:
    return Combine7(self.pa, self.pb, self.pc, self.pd,
                    const(None), const(None), const(None),
                    Fn[Tuple[A, B, C, D, None, None, None], E](
                      lambda t: self.abc((t[0], t[1], t[2], t[3]))))()

class Combine5(Generic[A, B, C, D, E, F]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], pe: Parser[E],
               abc: Callable[[Tuple[A, B, C, D, E]], F]) -> None:
    (self.pa, self.pb, self.pc, self.pd, self.pe, self.abc) = (pa, pb, pc, pd, pe, abc)

  def __call__(self) -> Parser[F]:
    return Combine7(self.pa, self.pb, self.pc, self.pd,
                    self.pe, const(None), const(None),
                    Fn[Tuple[A, B, C, D, E, None, None], F](
                      lambda t: self.abc((t[0], t[1], t[2], t[3], t[4]))))()

class Combine6(Generic[A, B, C, D, E, F, G]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], pe: Parser[E], pf: Parser[F],
               abc: Callable[[Tuple[A, B, C, D, E, F]], G]) -> None:
    (self.pa, self.pb, self.pc, self.pd, 
      self.pe, self.pf, self.abc) = (pa, pb, pc, pd, pe, pf, abc)

  def __call__(self) -> Parser[G]:
    return Combine7(self.pa, self.pb, self.pc, self.pd,
                    self.pe, self.pf, const(None),
                    Fn[Tuple[A, B, C, D, E, F, None], G](
                      lambda t: self.abc((t[0], t[1], t[2], t[3], t[4], t[5]))))()

errAcc = ListSg[ParseError]()

class Combine7(Generic[A, B, C, D, E, F, G, H]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], pe: Parser[E], pf: Parser[F], pg: Parser[G],
               abc: Callable[[Tuple[A, B, C, D, E, F, G]], H]) -> None:
    (self.pa, self.pb, self.pc, self.pd) = (pa, pb, pc, pd)
    (self.pe, self.pf, self.pg, self.abc) = (pe, pf, pg, abc)
    
  def __call__(self) -> Parser[H]:
    def x(j: JsonType) -> PreParsed[H]:
      fg = map2(append2sg(self.pf.run(j), self.pg.run(j), errAcc), lambda x: x)
      efg = map2(append2sg(self.pe.run(j), fg, errAcc), lambda x: (x[0],) +  x[1])
      defg = map2(append2sg(self.pd.run(j), efg, errAcc), lambda x: (x[0],) + x[1])
      cdefg = map2(append2sg(self.pc.run(j), defg, errAcc), lambda x: (x[0],) + x[1])
      bcdefg = map2(append2sg(self.pb.run(j), cdefg, errAcc), lambda x: (x[0],) + x[1])
      abcdefg = map2(append2sg(self.pa.run(j), bcdefg, errAcc), lambda x: (x[0],) + x[1])
      return toPreParsed[H]()(map2(abcdefg, self.abc))
    return Parser([], x)

def traverse(k: List[str], tk: List[str] = []) -> Callable[[JsonType], Parsed[JsonType]]:
  def x(j: JsonType) -> Parsed[JsonType]:
    if len(k) == 0:
      return F2(j)
    else:
      d = parseDict()(j)
      try:
        aj = map2(d, lambda y: y.get(k[0])) # type: ignore
        ntk = tk
        ntk.append(k[0])
        pj = toParsed[JsonType](ntk)(bind2(aj, parse_json)) # type: ignore
        return bind2(pj, traverse(k[1:], ntk))
      except Exception as e:
        return F1([ParseError(ntk, e)])
  return x
