from abc import ABC, abstractmethod
from adt import append2sg, bind2, fold2, fold3, fold4, map2, Sum2, Sum3, Sum4
from adt import Compose, Id, F1, F2, F3, F4, Fn, KeepLeft, ListSg, ProductSg, Semigroup
from dataclasses import dataclass
from functools import reduce
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

class ParseException(Exception): pass

@dataclass
class ParseTypeError(ParseException):
  expected: str
  encountered: str

@dataclass
class UnknownParseError(ParseException):
  origin: Exception

@dataclass
class CustomParseError(ParseException):
  aux: Dict[str, str]

ParseError = NamedTuple('ParseError', [('path', List[str]), ('error', ParseException)])
Parsed = Sum2[List[ParseError], A]
PreParseError = Sum2[List[ParseError], ParseException]
PreParsed = Sum2[PreParseError, A]
ParseFn = Callable[[JsonType], PreParsed[A]]

# work around mypy#731: no recursive structural types yet
class JsonList(List[JsonType]):
    pass

class JsonDict(Dict[str, JsonType]):
    pass

def load_json(j: str) -> PreParsed[any]: # type: ignore
  try: 
    return F2(json.loads(j)) # type: ignore
  except Exception as e:
    return F1(F2(UnknownParseError(e)))

def parse_json(x: any) -> PreParsed[JsonType]: # type: ignore
  try:
    if isinstance(x, list): # type: ignore
      return F2(F2(JsonList(x)))
    elif isinstance(x, dict): # type: ignore
      return F2(F3(JsonDict(x)))
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
        return F1(F2(ParseException))
  except Exception as e:
    return F1(F2(UnknownParseError(e)))

def error(t: Type[A]) -> Callable[[str], PreParseError]:
  return Fn[str, PreParseError](
    lambda s: F2(ParseTypeError(t.__name__, s)))

pathConcat = ListSg[str]()
class toParsed(Generic[A]):
  def __init__(self, path: List[str], sg: Semigroup[List[str]] = pathConcat) -> None:
    (self.path, self.sg) = (path, sg)
    
  def __call__(self, a: PreParsed[A]) -> Parsed[A]:
    return fold2[PreParseError, A, Parsed[A]](
      (fold2[List[ParseError], ParseException, Parsed[A]](
        (lambda x: F1(list(map(Fn[ParseError, ParseError](
          lambda y: y._replace(path=self.sg.append(self.path, y.path))), x))),
         lambda x: F1([ParseError(self.path, x)]))),
       lambda x: F2(x)))(a)

class toPreParsed(Generic[A]):
  def __call__(self, a: Parsed[A]) -> PreParsed[A]:
    return fold2[List[ParseError], A, PreParsed[A]](
      (lambda x: F1(F1(x)), lambda x: F2(x)))(a)

class Parser(Generic[A]):
  
  def __init__(self, path: List[str], run: ParseFn[A]) -> None:
    self._path = path
    self._run = run

  def run(self, j: JsonType) -> Parsed[A]: 
    return bind2(traverse(self._path)(j), Compose(toParsed[A](self._path), self._run))

  def setPath(self, path: List[str]) -> 'Parser[A]':
    return self.mapPath(lambda _: path)

  def mapPath(self, update: Callable[[List[str]], List[str]]) -> 'Parser[A]':
    return Parser(update(self._path), self._run)

  def parse(self, s: str) -> Parsed[A]:
    return bind2(toParsed[JsonType](self._path)(
            bind2(load_json(s), parse_json)), self.run)

listAcc = ProductSg(ListSg[ParseError](), ListSg[A]())

ListResult = Tuple[List[ParseError], List[A]]
replacePath = KeepLeft[List[str]]()
class ListParser:
  class _Base(ABC, Generic[A, B], Parser[B]):

    def __init__(self, path: List[str], run: ParseFn[A]) -> None:
      self._path = path
      self._runList = run

    @abstractmethod
    def bind(self, l: ListResult[A]) -> Parsed[B]: pass
    def run(self, j: JsonType) -> Parsed[B]:
      def reducer(l: JsonList) -> Parsed[B]:
        def x(acc: Tuple[int, ListResult[A]], j: JsonType) -> Tuple[int, ListResult[A]]:
          return (acc[0] + 1,
            listAcc.append(acc[1],
              fold2[List[ParseError], A, ListResult[A]](
                (lambda x: (x, []),
                 lambda x: ([], [x])))(
                  toParsed[A](self._path + [str(acc[0])], replacePath)(
                    bind2(parse_json(j), self._runList)))))
        return self.bind(reduce(x, l, (0, Id[ListResult[A]]()(([], []))))[1])
      return bind2(Parser(self._path, parseList()).run(j), reducer)

  class FailFast(Generic[A], _Base[A, List[A]]):

    def bind(self, r: ListResult[A]) -> Parsed[List[A]]:
      return F1(r[0]) if (not (len(r[0]) == 0)) else F2(r[1])

  class FailSlow(Generic[A], _Base[A, ListResult[A]]):

    def bind(self, r: ListResult[A]) -> Parsed[ListResult[A]]:
      return F1(r[0]) if (len(r[1]) == 0 and not (len(r[0]) == 0)) else F2(r)

def parseDict() -> ParseFn[JsonDict]:
  err = error(JsonDict)
  return fold3[JsonPrimitive, 'JsonList', 'JsonDict', PreParsed[JsonDict]](
    (lambda _: F1(err('JsonPrimitive')),
     lambda _: F1(err('List')),
     lambda x: F2(x)))

def parseList() -> ParseFn[JsonList]:
  err = error(JsonList)
  return fold3[JsonPrimitive, 'JsonList', 'JsonDict', PreParsed[JsonList]](
    (lambda _: F1(err('JsonPrimitive')),
     lambda x: F2(x),
     lambda _: F1(err('Dict'))))

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

def parseStr(j: JsonType) -> PreParsed[str]:
  (_, b, c, d) = prims(str).fold
  prim = fold4[str, int, bool, None, PreParsed[str]]((lambda x: F2(x), b, c, d))
  return parsePrim(str, prim)(j)

def parseInt(j: JsonType) -> PreParsed[int]:
  (a, _, c, d) = prims(int).fold
  prim = fold4[str, int, bool, None, PreParsed[int]]((a, lambda x: F2(x), c, d))
  return parsePrim(int, prim)(j)

def parseBool(j: JsonType) -> PreParsed[bool]:
  (a, b, _, d) = prims(bool).fold
  prim = fold4[str, int, bool, None, PreParsed[bool]]((a, b, lambda x: F2(x), d))
  return parsePrim(bool, prim)(j)

def parseNone(j: JsonType) -> PreParsed[None]:
  (a, b, c, _) = prims(type(None)).fold
  prim = fold4[str, int, bool, None, PreParsed[None]]((a, b, c, lambda x: F2(x)))
  return parsePrim(type(None), prim)(j)

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
           lambda x: F2(F1(x))))(parseNone(j)),
       lambda x: F2(F2(x))))(p(j)))

class extendParse(Generic[A, B]):
  
  def __init__(self, p: ParseFn[A], c: Callable[[A], Sum2[CustomParseError, B]]) -> None:
    (self._p, self._c) = (p, c)

  def __call__(self, j: JsonType) -> PreParsed[B]:
    return Fn[JsonType, PreParsed[B]](
      lambda x: bind2(self._p(x), 
        Compose(fold2[CustomParseError, B, PreParsed[B]](
          (lambda x: F1(F2(x)), lambda x: F2(x))), self._c)))(j)

errAcc = ListSg[ParseError]()

class Parse7(Generic[A, B, C, D, E, F, G, H], Parser[H]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], pe: Parser[E], pf: Parser[F], pg: Parser[G],
               abc: Callable[[Tuple[A, B, C, D, E, F, G]], H]) -> None:
    (self.pa, self.pb, self.pc, self.pd) = (pa, pb, pc, pd)
    (self.pe, self.pf, self.pg, self.abc) = (pe, pf, pg, abc)
    def run(j: JsonType) -> PreParsed[H]:
      fg = map2(append2sg(self.pf.run(j), self.pg.run(j), errAcc), lambda x: x)
      efg = map2(append2sg(self.pe.run(j), fg, errAcc), lambda x: (x[0],) +  x[1])
      defg = map2(append2sg(self.pd.run(j), efg, errAcc), lambda x: (x[0],) + x[1])
      cdefg = map2(append2sg(self.pc.run(j), defg, errAcc), lambda x: (x[0],) + x[1])
      bcdefg = map2(append2sg(self.pb.run(j), cdefg, errAcc), lambda x: (x[0],) + x[1])
      abcdefg = map2(append2sg(self.pa.run(j), bcdefg, errAcc), lambda x: (x[0],) + x[1])
      return toPreParsed[H]()(map2(abcdefg, self.abc))
    self._run = run
    self._path = []

class Parse1(Generic[A, B], Parse7[A, None, None, None, None, None, None, B]):

  def __init__(self, pa: Parser[A], ab: Callable[[A], B]) -> None:
    super().__init__(pa, const(None), const(None), const(None), 
                   const(None), const(None), const(None), 
                   Fn[Tuple[A, None, None, None, None, None, None], B](
                    lambda t: ab(t[0])))

class Parse2(Generic[A, B, C], Parse7[A, B, None, None, None, None, None, C]):

  def __init__(self, pa: Parser[A], pb: Parser[B], abc: Callable[[Tuple[A, B]], C]) -> None:
    super().__init__(pa, pb, const(None), const(None), 
                   const(None), const(None), const(None), 
                   Fn[Tuple[A, B, None, None, None, None, None], C](
                    lambda t: abc((t[0], t[1]))))

class Parse3(Generic[A, B, C, D], Parse7[A, B, C, None, None, None, None, D]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               abc: Callable[[Tuple[A, B, C]], D]) -> None:
    super().__init__(pa, pb, pc, const(None),
                   const(None), const(None), const(None), 
                   Fn[Tuple[A, B, C, None, None, None, None], D](
                    lambda t: abc((t[0], t[1], t[2]))))

class Parse4(Generic[A, B, C, D, E], Parse7[A, B, C, D, None, None, None, E]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], abc: Callable[[Tuple[A, B, C, D]], E]) -> None:
    super().__init__(pa, pb, pc, pd,
                   const(None), const(None), const(None), 
                   Fn[Tuple[A, B, C, D, None, None, None], E](
                    lambda t: abc((t[0], t[1], t[2], t[3]))))

class Parse5(Generic[A, B, C, D, E, F], Parse7[A, B, C, D, E, None, None, F]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], pe: Parser[E],
               abc: Callable[[Tuple[A, B, C, D, E]], F]) -> None:
    super().__init__(pa, pb, pc, pd, pe,
                   const(None), const(None), 
                   Fn[Tuple[A, B, C, D, E, None, None], F](
                    lambda t: abc((t[0], t[1], t[2], t[3], t[4]))))

class Parse6(Generic[A, B, C, D, E, F, G], Parse7[A, B, C, D, E, F, None, G]):

  def __init__(self, pa: Parser[A], pb: Parser[B], pc: Parser[C],
               pd: Parser[D], pe: Parser[E], pf: Parser[F],
               abc: Callable[[Tuple[A, B, C, D, E, F]], G]) -> None:
    super().__init__(pa, pb, pc, pd, pe, pf, const(None), 
                   Fn[Tuple[A, B, C, D, E, F, None], G](
                    lambda t: abc((t[0], t[1], t[2], t[3], t[4], t[5]))))

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
        return F1([ParseError(ntk, UnknownParseError(e))])
  return x
