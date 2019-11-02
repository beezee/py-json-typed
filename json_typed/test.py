from adt import Id, Sum2, F1, F2, Fn, map2
from dataclasses import dataclass
from json_typed import CustomParseError, ExtendParse
from json_typed import Parse2, Parse3, ListParser, ParseError, parse_str
from json_typed import parse_all_list, parse_any_list, parse_int, parse_optional, Parser
from json_typed import ExtendSerialize, Serialize2, Serialize3, ListSerializer
from json_typed import serialize_str, serialize_int, serialize_optional
from json_typed import serialize_list, Serializer
from typing import List, Tuple
import json

@dataclass 
class FooBaz:
  foo: Sum2[None, str]
  bar: List[int]
  baz: Tuple[List[ParseError], List[str]]
@dataclass
class FBBQ:
  foobar: Tuple[List[ParseError], List[FooBaz]]
  quux: int

foobaz = Parse3(FooBaz)(
  parse_optional(parse_str(['foo'])),
  parse_all_list(parse_int(['bar'])),
  parse_any_list(parse_str(['baz'])))
fbq = Parse2(FBBQ)(
  parse_any_list(foobaz).set_path(['foobar']),
  parse_int(['quux']))
  
sfbq = Serialize2(FBBQ, lambda x: (x.foobar[1], x.quux))(
  serialize_list(Serialize3(FooBaz, lambda x: (x.foo, x.bar, x.baz[1]), ['foobar'])(
    serialize_optional(serialize_str('foo')),
    serialize_list(serialize_int('bar')),
    serialize_list(serialize_str('baz')))),
  serialize_int('quux'))

if __name__ == '__main__':
  print(fbq.parse('{"quux": null, "foobar": [{"foo": "bar", "bar": [], "baz": ["quux"]}]}'))
  print(fbq.parse('{"quux": -4, "foobar": [1, {"foo": "bar", "bar": [], "baz": [1, "quux"]}]}'))
  print(fbq.parse('{"quux": 33, "foobar": [{"foo": null, "bar": ["a", 2, "3"]}]}'))
  print(fbq.parse('{"quux": null, "foobar": [{"foo": 4, "bar": 3, "baz": [2, "quux", 3]}]}'))
  print(fbq.parse('{"quux": -5, "foobar": [{"bar": [2, 3, 4], "moo": "bar", "baz": []}]}'))

  print(
    map2(fbq.parse('{"quux": -4, "foobar": [1, {"foo": "bar", "bar": [], "baz": [1, "quux"]}]}'),
    sfbq.serialize))
